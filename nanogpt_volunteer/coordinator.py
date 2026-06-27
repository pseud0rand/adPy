"""
Coordinator server for volunteer-computed nanoGPT training.

The coordinator holds the authoritative model state, distributes data batches
to connected volunteers, collects and aggregates their gradients, and applies
optimizer updates. Training proceeds in rounds: each round the coordinator
sends a batch to every available volunteer, waits for gradients (with a
configurable timeout), averages them, and steps the optimizer.

Usage:
    python -m nanogpt_volunteer.coordinator --data_dir data/shakespeare_char --out_dir out-volunteer
"""

import os
import sys
import time
import math
import asyncio
import pickle
import logging
import argparse
import signal
from dataclasses import dataclass, asdict
from typing import Dict, Optional
from contextlib import nullcontext

import numpy as np
import torch

from .model import GPT, GPTConfig
from .protocol import (
    MsgType, send_msg, recv_msg,
    _serialize_state_dict, _deserialize_state_dict,
    _serialize_tensors, _deserialize_tensors,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [COORD] %(message)s')
log = logging.getLogger(__name__)


@dataclass
class VolunteerInfo:
    volunteer_id: str
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    device: str = 'cpu'
    busy: bool = False
    batches_completed: int = 0
    last_heartbeat: float = 0.0
    addr: str = ''


class Coordinator:

    def __init__(self, args):
        self.args = args
        self.volunteers: Dict[str, VolunteerInfo] = {}
        self._next_vol_id = 0
        self._shutdown = False

        self.device = args.device
        self.device_type = 'cuda' if 'cuda' in self.device else 'cpu'

        self._init_data()
        self._init_model()

        self.iter_num = 0
        self.best_val_loss = 1e9

        self._pending_gradients: list = []
        self._gradient_event = asyncio.Event()

    def _init_data(self):
        data_dir = self.args.data_dir
        self.train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        val_path = os.path.join(data_dir, 'val.bin')
        self.val_data = np.memmap(val_path, dtype=np.uint16, mode='r') if os.path.exists(val_path) else None

        meta_path = os.path.join(data_dir, 'meta.pkl')
        self.meta_vocab_size = None
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            self.meta_vocab_size = meta['vocab_size']
            log.info("Found vocab_size = %d in %s", self.meta_vocab_size, meta_path)

    def _init_model(self):
        args = self.args
        vocab_size = self.meta_vocab_size if self.meta_vocab_size is not None else 50304

        if args.init_from == 'resume' and os.path.exists(os.path.join(args.out_dir, 'ckpt.pt')):
            log.info("Resuming from checkpoint in %s", args.out_dir)
            ckpt = torch.load(os.path.join(args.out_dir, 'ckpt.pt'), map_location=self.device)
            conf = GPTConfig(**ckpt['model_args'])
            self.model = GPT(conf)
            state_dict = ckpt['model']
            unwanted_prefix = '_orig_mod.'
            for k in list(state_dict.keys()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            self.model.load_state_dict(state_dict)
            self.iter_num = ckpt.get('iter_num', 0)
            self.best_val_loss = ckpt.get('best_val_loss', 1e9)
        else:
            log.info("Initializing new model from scratch")
            conf = GPTConfig(
                block_size=args.block_size,
                vocab_size=vocab_size,
                n_layer=args.n_layer,
                n_head=args.n_head,
                n_embd=args.n_embd,
                dropout=args.dropout,
                bias=args.bias,
            )
            self.model = GPT(conf)

        self.model.to(self.device)
        log.info("Model has %.2fM parameters", self.model.get_num_params() / 1e6)

        self.optimizer = self.model.configure_optimizers(
            args.weight_decay, args.learning_rate, (args.beta1, args.beta2), self.device_type
        )

        self.ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
        self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(device_type=self.device_type, dtype=self.ptdtype)
        self.scaler = torch.amp.GradScaler(enabled=(args.dtype == 'float16'))

    def get_batch(self, split='train'):
        data = self.train_data if split == 'train' else self.val_data
        if data is None:
            return None, None
        ix = torch.randint(len(data) - self.args.block_size, (self.args.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i + self.args.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + self.args.block_size]).astype(np.int64)) for i in ix])
        return x, y

    def get_lr(self, it):
        args = self.args
        if not args.decay_lr:
            return args.learning_rate
        if it < args.warmup_iters:
            return args.learning_rate * (it + 1) / (args.warmup_iters + 1)
        if it > args.lr_decay_iters:
            return args.min_lr
        decay_ratio = (it - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return args.min_lr + coeff * (args.learning_rate - args.min_lr)

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            if split == 'val' and self.val_data is None:
                continue
            losses = torch.zeros(self.args.eval_iters)
            for k in range(self.args.eval_iters):
                X, Y = self.get_batch(split)
                X, Y = X.to(self.device), Y.to(self.device)
                with self.ctx:
                    _, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def save_checkpoint(self):
        os.makedirs(self.args.out_dir, exist_ok=True)
        raw_model = self.model
        checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'model_args': asdict(raw_model.config),
            'iter_num': self.iter_num,
            'best_val_loss': self.best_val_loss,
        }
        path = os.path.join(self.args.out_dir, 'ckpt.pt')
        torch.save(checkpoint, path)
        log.info("Saved checkpoint to %s", path)

    def _validate_gradients(self, grads: dict) -> bool:
        for name, g in grads.items():
            if torch.isnan(g).any() or torch.isinf(g).any():
                return False
        return True

    def _aggregate_gradients(self, gradient_list: list):
        n = len(gradient_list)
        if n == 0:
            return
        ref_keys = set(gradient_list[0].keys())
        aggregated = {}
        for key in ref_keys:
            stacked = torch.stack([g[key].to(self.device) for g in gradient_list if key in g])
            aggregated[key] = stacked.mean(dim=0)

        self.optimizer.zero_grad(set_to_none=True)
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in aggregated:
                param.grad = aggregated[name]

        if self.args.grad_clip != 0.0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)

        self.scaler.step(self.optimizer)
        self.scaler.update()

    async def _handle_volunteer(self, reader, writer):
        addr = writer.get_extra_info('peername')
        vol_id = f"vol-{self._next_vol_id}"
        self._next_vol_id += 1
        log.info("New connection from %s, assigned %s", addr, vol_id)

        vol = VolunteerInfo(
            volunteer_id=vol_id,
            reader=reader,
            writer=writer,
            last_heartbeat=time.time(),
            addr=str(addr),
        )

        try:
            msg_type, metadata, _ = await asyncio.wait_for(recv_msg(reader), timeout=30)
            if msg_type != MsgType.REGISTER:
                log.warning("%s: expected REGISTER, got %s", vol_id, msg_type)
                return
            vol.device = metadata.get('device', 'cpu')
            log.info("%s registered: device=%s", vol_id, vol.device)
        except (asyncio.TimeoutError, Exception) as e:
            log.warning("%s: registration failed: %s", vol_id, e)
            return

        self.volunteers[vol_id] = vol

        state_bytes = _serialize_state_dict(self.model.state_dict())
        config_dict = asdict(self.model.config)
        await send_msg(writer, MsgType.WELCOME, {
            'volunteer_id': vol_id,
            'config': config_dict,
            'iter_num': self.iter_num,
        }, state_bytes)
        log.info("%s: sent WELCOME with model state (%d bytes compressed)", vol_id, len(state_bytes))

        try:
            while not self._shutdown:
                try:
                    msg_type, metadata, tensor_payload = await asyncio.wait_for(
                        recv_msg(reader), timeout=120
                    )
                except asyncio.TimeoutError:
                    await send_msg(writer, MsgType.HEARTBEAT_PING, {})
                    try:
                        msg_type, metadata, _ = await asyncio.wait_for(recv_msg(reader), timeout=30)
                        if msg_type == MsgType.HEARTBEAT_PONG:
                            vol.last_heartbeat = time.time()
                            continue
                    except (asyncio.TimeoutError, Exception):
                        log.warning("%s: heartbeat timeout, disconnecting", vol_id)
                        break
                    continue

                if msg_type == MsgType.GRADIENTS:
                    grads = _deserialize_tensors(tensor_payload)
                    if self._validate_gradients(grads):
                        vol.batches_completed += 1
                        loss_val = metadata.get('loss', None)
                        compute_time = metadata.get('compute_time', None)
                        log.info(
                            "%s: received gradients (batch %d, loss=%.4f, compute=%.2fs)",
                            vol_id, vol.batches_completed,
                            loss_val if loss_val else -1,
                            compute_time if compute_time else -1,
                        )
                        self._pending_gradients.append(grads)
                        self._gradient_event.set()
                    else:
                        log.warning("%s: received invalid gradients (NaN/Inf), discarding", vol_id)

                elif msg_type == MsgType.HEARTBEAT_PONG:
                    vol.last_heartbeat = time.time()

                elif msg_type == MsgType.STATUS:
                    log.info("%s: status update: %s", vol_id, metadata)

        except (ConnectionResetError, asyncio.IncompleteReadError, BrokenPipeError) as e:
            log.info("%s: disconnected (%s)", vol_id, type(e).__name__)
        finally:
            self.volunteers.pop(vol_id, None)
            writer.close()
            log.info("%s: cleaned up. %d volunteers remaining", vol_id, len(self.volunteers))

    async def _distribute_batch(self):
        x, y = self.get_batch('train')
        if x is None:
            return

        batch_tensors = _serialize_tensors({'x': x, 'y': y})
        lr = self.get_lr(self.iter_num)

        disconnected = []
        for vol_id, vol in self.volunteers.items():
            try:
                await send_msg(vol.writer, MsgType.BATCH, {
                    'iter_num': self.iter_num,
                    'learning_rate': lr,
                }, batch_tensors)
                vol.busy = True
            except (ConnectionResetError, BrokenPipeError, OSError):
                disconnected.append(vol_id)

        for vol_id in disconnected:
            self.volunteers.pop(vol_id, None)
            log.warning("Lost connection to %s during batch distribution", vol_id)

    async def _broadcast_weights(self):
        state_bytes = _serialize_state_dict(self.model.state_dict())

        disconnected = []
        for vol_id, vol in self.volunteers.items():
            try:
                await send_msg(vol.writer, MsgType.WEIGHT_UPDATE, {
                    'iter_num': self.iter_num,
                }, state_bytes)
                vol.busy = False
            except (ConnectionResetError, BrokenPipeError, OSError):
                disconnected.append(vol_id)

        for vol_id in disconnected:
            self.volunteers.pop(vol_id, None)

    async def _training_loop(self):
        args = self.args
        log.info("Training loop started. Waiting for volunteers...")

        while not self._shutdown and self.iter_num <= args.max_iters:
            while len(self.volunteers) == 0 and not self._shutdown:
                await asyncio.sleep(1)

            if self._shutdown:
                break

            if self.iter_num % args.eval_interval == 0:
                losses = self.estimate_loss()
                train_loss = losses.get('train', float('nan'))
                val_loss = losses.get('val', float('nan'))
                log.info(
                    "step %d: train loss %.4f, val loss %.4f, volunteers: %d",
                    self.iter_num, train_loss, val_loss, len(self.volunteers),
                )
                if val_loss < self.best_val_loss or args.always_save_checkpoint:
                    self.best_val_loss = val_loss if not math.isnan(val_loss) else self.best_val_loss
                    if self.iter_num > 0:
                        self.save_checkpoint()

            lr = self.get_lr(self.iter_num)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            n_volunteers = len(self.volunteers)
            if n_volunteers == 0:
                continue

            self._pending_gradients.clear()
            self._gradient_event.clear()

            await self._distribute_batch()

            min_responses = max(1, min(n_volunteers, args.min_volunteer_responses))
            deadline = time.time() + args.volunteer_timeout

            while len(self._pending_gradients) < min_responses:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                self._gradient_event.clear()
                try:
                    await asyncio.wait_for(self._gradient_event.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    break

            if len(self._pending_gradients) == 0:
                log.warning("No gradients received for iter %d, skipping", self.iter_num)
                continue

            log.info(
                "iter %d: aggregating gradients from %d/%d volunteers, lr=%.6f",
                self.iter_num, len(self._pending_gradients), n_volunteers, lr,
            )

            self._aggregate_gradients(self._pending_gradients)

            await self._broadcast_weights()

            self.iter_num += 1

            if self.iter_num % args.log_interval == 0:
                log.info(
                    "iter %d complete, %d volunteers connected",
                    self.iter_num, len(self.volunteers),
                )

        log.info("Training complete at iter %d", self.iter_num)
        self.save_checkpoint()

    async def _shutdown_volunteers(self):
        for vol_id, vol in list(self.volunteers.items()):
            try:
                await send_msg(vol.writer, MsgType.SHUTDOWN, {})
                vol.writer.close()
            except Exception:
                pass
        self.volunteers.clear()

    async def run(self):
        server = await asyncio.start_server(
            self._handle_volunteer, self.args.host, self.args.port,
        )
        addr = server.sockets[0].getsockname()
        log.info("Coordinator listening on %s:%d", addr[0], addr[1])
        log.info("Volunteers can connect with: python -m nanogpt_volunteer.volunteer --host %s --port %d", addr[0], addr[1])

        loop = asyncio.get_event_loop()

        def signal_handler():
            log.info("Shutdown signal received")
            self._shutdown = True

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)

        training_task = asyncio.create_task(self._training_loop())

        async with server:
            try:
                await asyncio.gather(
                    server.serve_forever(),
                    training_task,
                    return_exceptions=True,
                )
            except asyncio.CancelledError:
                pass

        await self._shutdown_volunteers()
        log.info("Coordinator shut down")


def parse_args():
    p = argparse.ArgumentParser(description='nanoGPT Volunteer Computing - Coordinator')
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--port', type=int, default=9876)
    p.add_argument('--data_dir', default='data/shakespeare_char')
    p.add_argument('--out_dir', default='out-volunteer')
    p.add_argument('--init_from', default='scratch', choices=['scratch', 'resume'])

    p.add_argument('--batch_size', type=int, default=12)
    p.add_argument('--block_size', type=int, default=256)
    p.add_argument('--n_layer', type=int, default=6)
    p.add_argument('--n_head', type=int, default=6)
    p.add_argument('--n_embd', type=int, default=384)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--bias', type=bool, default=False)

    p.add_argument('--learning_rate', type=float, default=1e-3)
    p.add_argument('--max_iters', type=int, default=5000)
    p.add_argument('--weight_decay', type=float, default=1e-1)
    p.add_argument('--beta1', type=float, default=0.9)
    p.add_argument('--beta2', type=float, default=0.99)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--decay_lr', type=bool, default=True)
    p.add_argument('--warmup_iters', type=int, default=100)
    p.add_argument('--lr_decay_iters', type=int, default=5000)
    p.add_argument('--min_lr', type=float, default=1e-4)

    p.add_argument('--eval_interval', type=int, default=250)
    p.add_argument('--eval_iters', type=int, default=20)
    p.add_argument('--log_interval', type=int, default=10)
    p.add_argument('--always_save_checkpoint', type=bool, default=True)

    p.add_argument('--dtype', default='float32', choices=['float32', 'bfloat16', 'float16'])
    p.add_argument('--device', default='cpu')

    p.add_argument('--min_volunteer_responses', type=int, default=1,
                   help='Minimum volunteer gradient responses before aggregating')
    p.add_argument('--volunteer_timeout', type=float, default=120.0,
                   help='Seconds to wait for volunteer gradients per round')

    return p.parse_args()


def main():
    args = parse_args()
    coordinator = Coordinator(args)
    asyncio.run(coordinator.run())


if __name__ == '__main__':
    main()
