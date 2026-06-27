"""
Volunteer client for nanoGPT volunteer computing.

Connects to a coordinator server, receives model weights and data batches,
computes forward/backward passes locally, and sends gradients back.
Volunteers contribute their CPU/GPU resources to collaborative training.

Usage:
    python -m nanogpt_volunteer.volunteer --host 192.168.1.100 --port 9876
    python -m nanogpt_volunteer.volunteer --host coordinator.example.com --device cuda
"""

import os
import sys
import time
import asyncio
import logging
import argparse
import signal
from dataclasses import asdict
from contextlib import nullcontext

import torch

from .model import GPT, GPTConfig
from .protocol import (
    MsgType, send_msg, recv_msg,
    _serialize_state_dict, _deserialize_state_dict,
    _serialize_tensors, _deserialize_tensors,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [VOLUNTEER] %(message)s')
log = logging.getLogger(__name__)


class Volunteer:

    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.device_type = 'cuda' if 'cuda' in self.device else 'cpu'
        self.model = None
        self.volunteer_id = None
        self._shutdown = False
        self.batches_computed = 0

        self.ptdtype = {
            'float32': torch.float32,
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
        }[args.dtype]
        self.ctx = (
            nullcontext() if self.device_type == 'cpu'
            else torch.amp.autocast(device_type=self.device_type, dtype=self.ptdtype)
        )
        self.scaler = torch.amp.GradScaler(enabled=(args.dtype == 'float16'))

    def _init_model(self, config_dict, state_bytes):
        config = GPTConfig(**config_dict)
        self.model = GPT(config)
        state_dict = _deserialize_state_dict(state_bytes)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.train()
        log.info(
            "Model initialized: %.2fM parameters on %s",
            self.model.get_num_params() / 1e6, self.device,
        )

    def _update_weights(self, state_bytes):
        state_dict = _deserialize_state_dict(state_bytes)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

    def _compute_gradients(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)

        self.model.zero_grad(set_to_none=True)

        t0 = time.time()
        with self.ctx:
            logits, loss = self.model(x, y)

        self.scaler.scale(loss).backward()

        if self.args.dtype == 'float16':
            self.scaler.unscale_(
                torch.optim.SGD(self.model.parameters(), lr=0)
            )
            self.scaler.update()

        compute_time = time.time() - t0

        gradients = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                gradients[name] = param.grad.cpu().clone()

        return gradients, loss.item(), compute_time

    async def _connect_with_retry(self):
        attempt = 0
        max_attempts = self.args.max_reconnect_attempts
        while not self._shutdown:
            try:
                reader, writer = await asyncio.open_connection(
                    self.args.host, self.args.port,
                )
                log.info("Connected to coordinator at %s:%d", self.args.host, self.args.port)
                return reader, writer
            except (ConnectionRefusedError, OSError) as e:
                attempt += 1
                if max_attempts > 0 and attempt >= max_attempts:
                    log.error("Max reconnect attempts reached, giving up")
                    return None, None
                wait = min(2 ** attempt, 60)
                log.warning("Connection failed (%s), retrying in %ds (attempt %d)...", e, wait, attempt)
                await asyncio.sleep(wait)
        return None, None

    async def run(self):
        loop = asyncio.get_event_loop()

        def signal_handler():
            log.info("Shutdown signal received")
            self._shutdown = True

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)

        while not self._shutdown:
            reader, writer = await self._connect_with_retry()
            if reader is None:
                break

            try:
                await self._session(reader, writer)
            except (ConnectionResetError, asyncio.IncompleteReadError, BrokenPipeError) as e:
                log.warning("Disconnected from coordinator: %s", e)
            finally:
                writer.close()

            if not self._shutdown and self.args.auto_reconnect:
                log.info("Will attempt to reconnect...")
                await asyncio.sleep(5)
            else:
                break

        log.info("Volunteer shutting down. Computed %d batches total.", self.batches_computed)

    async def _session(self, reader, writer):
        await send_msg(writer, MsgType.REGISTER, {
            'device': self.device,
            'dtype': self.args.dtype,
            'has_cuda': torch.cuda.is_available(),
            'cuda_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        })

        msg_type, metadata, tensor_payload = await asyncio.wait_for(
            recv_msg(reader), timeout=60,
        )
        if msg_type != MsgType.WELCOME:
            log.error("Expected WELCOME, got %s", msg_type)
            return

        self.volunteer_id = metadata['volunteer_id']
        config_dict = metadata['config']
        iter_num = metadata['iter_num']
        log.info("Registered as %s, starting from iter %d", self.volunteer_id, iter_num)

        self._init_model(config_dict, tensor_payload)

        while not self._shutdown:
            try:
                msg_type, metadata, tensor_payload = await asyncio.wait_for(
                    recv_msg(reader), timeout=180,
                )
            except asyncio.TimeoutError:
                log.warning("No message from coordinator for 180s, disconnecting")
                break

            if msg_type == MsgType.BATCH:
                iter_num = metadata.get('iter_num', -1)
                batch_data = _deserialize_tensors(tensor_payload)
                x, y = batch_data['x'], batch_data['y']

                log.info("Computing gradients for iter %d (batch shape: %s)", iter_num, list(x.shape))
                gradients, loss, compute_time = self._compute_gradients(x, y)

                grad_bytes = _serialize_tensors(gradients)
                await send_msg(writer, MsgType.GRADIENTS, {
                    'iter_num': iter_num,
                    'loss': loss,
                    'compute_time': compute_time,
                    'volunteer_id': self.volunteer_id,
                }, grad_bytes)

                self.batches_computed += 1
                log.info(
                    "Sent gradients for iter %d (loss=%.4f, compute=%.2fs, total batches=%d)",
                    iter_num, loss, compute_time, self.batches_computed,
                )

            elif msg_type == MsgType.WEIGHT_UPDATE:
                self._update_weights(tensor_payload)
                log.info("Weights updated to iter %d", metadata.get('iter_num', -1))

            elif msg_type == MsgType.HEARTBEAT_PING:
                await send_msg(writer, MsgType.HEARTBEAT_PONG, {})

            elif msg_type == MsgType.SHUTDOWN:
                log.info("Coordinator requested shutdown")
                break

            else:
                log.warning("Unknown message type: %s", msg_type)


def parse_args():
    p = argparse.ArgumentParser(description='nanoGPT Volunteer Computing - Volunteer Client')
    p.add_argument('--host', default='localhost',
                   help='Coordinator hostname or IP')
    p.add_argument('--port', type=int, default=9876,
                   help='Coordinator port')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                   help='Compute device (cpu, cuda, cuda:0, mps)')
    p.add_argument('--dtype', default='float32', choices=['float32', 'bfloat16', 'float16'],
                   help='Compute dtype')
    p.add_argument('--auto_reconnect', action='store_true', default=True,
                   help='Automatically reconnect on disconnection')
    p.add_argument('--max_reconnect_attempts', type=int, default=0,
                   help='Max reconnect attempts (0 = unlimited)')
    return p.parse_args()


def main():
    args = parse_args()
    volunteer = Volunteer(args)
    asyncio.run(volunteer.run())


if __name__ == '__main__':
    main()
