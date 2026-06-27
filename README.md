# adPy
*A Bare-bones Implementation of Automatic Differentiation in Python*

The Python scripts are simple implementations of some of the concepts discussed in "Introduction to Automatic Differentiation and MATLAB Object-Oriented Programming" by Richard D. Neidinger (https://doi.org/10.1137/080743627)

The scripts were written in the process of learning about automatic differentiation. They are based on operator overloading, and only require NumPy. 

The adpy_reverse_mode.py is the reverse form, containing the back-propagation step in which local derivatives are collected. This is done using dicionaries, and dynamic creation of key labels that correspond to symbolic representations of local derivatives. The print statements are dispersed in the script for diagnostic purposes, to track what is happening, and can be commented out. The performance has not been tested, but is likely low, given the method of implementation. The main purpose is educational.

Only a few special functions are implemented. More can be easily added at the end of each script, following the form of others.

---

## nanoGPT Volunteer Computing

Train GPT language models using volunteer-contributed compute resources. Based on [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT), this extends the training infrastructure with a distributed volunteer computing architecture where anyone can contribute CPU or GPU resources to collaboratively train a model.

### Architecture

The system uses a **parameter server** pattern with synchronous gradient aggregation:

- **Coordinator** — holds the authoritative model state, distributes data batches to connected volunteers, collects their computed gradients, aggregates them, and applies optimizer updates.
- **Volunteers** — connect to the coordinator over TCP, receive model weights and data batches, compute forward/backward passes locally, and send gradients back.
- Communication uses a length-prefixed binary protocol with msgpack metadata and zlib-compressed PyTorch tensor payloads.

```
┌─────────────────────────────────────────────────────┐
│                   COORDINATOR                       │
│  ┌───────────┐  ┌───────────┐  ┌─────────────────┐ │
│  │   Model    │  │ Optimizer │  │  Data Loader    │ │
│  │  State     │  │ (AdamW)   │  │ (train/val.bin) │ │
│  └─────┬─────┘  └─────┬─────┘  └────────┬────────┘ │
│        │              │                  │          │
│        └──────────┬───┘──────────────────┘          │
│                   │                                 │
│         ┌─────────▼──────────┐                      │
│         │ Gradient Aggregator│                      │
│         └─────────┬──────────┘                      │
└───────────────────┼─────────────────────────────────┘
                    │ TCP (port 9876)
          ┌─────────┼─────────┐
          │         │         │
    ┌─────▼───┐ ┌───▼─────┐ ┌▼────────┐
    │Volunteer│ │Volunteer│ │Volunteer│
    │ (CPU)   │ │ (GPU)   │ │ (CPU)   │
    └─────────┘ └─────────┘ └─────────┘
```

### Quick Start

```bash
pip install torch numpy msgpack

# 1. Prepare data (downloads tiny shakespeare by default)
python -m nanogpt_volunteer.prepare_data

# 2. Start coordinator (on a machine with the data)
python -m nanogpt_volunteer.coordinator --data_dir data/shakespeare_char

# 3. Connect volunteers (on any machine, as many as you like)
python -m nanogpt_volunteer.volunteer --host <coordinator-ip> --port 9876

# 4. Sample from trained model
python -m nanogpt_volunteer.sample --out_dir out-volunteer
```

### Coordinator Options

```
--host              Listen address (default: 0.0.0.0)
--port              Listen port (default: 9876)
--data_dir          Path to prepared data directory
--out_dir           Checkpoint output directory
--batch_size        Micro-batch size sent to each volunteer (default: 12)
--block_size        Context window length (default: 256)
--n_layer           Transformer layers (default: 6)
--n_head            Attention heads (default: 6)
--n_embd            Embedding dimension (default: 384)
--max_iters         Total training iterations (default: 5000)
--learning_rate     Peak learning rate (default: 1e-3)
--min_volunteer_responses  Min gradients before aggregating (default: 1)
--volunteer_timeout Seconds to wait per round (default: 120)
```

### Volunteer Options

```
--host              Coordinator hostname/IP (default: localhost)
--port              Coordinator port (default: 9876)
--device            Compute device: cpu, cuda, cuda:0, mps (auto-detected)
--dtype             float32, bfloat16, or float16 (default: float32)
--auto_reconnect    Reconnect on disconnection (default: true)
```

### How It Works

1. The coordinator initializes the model and loads training data.
2. Volunteers connect and receive the full model state.
3. Each training round: the coordinator sends a data batch to all connected volunteers.
4. Each volunteer computes forward + backward pass locally and sends gradients back.
5. The coordinator aggregates gradients (averaging across volunteers) and steps the optimizer.
6. Updated weights are broadcast to all volunteers.
7. Volunteers can join or leave at any time — training adapts to the available pool.

### Features

- **Fault tolerant** — volunteers can disconnect/reconnect at any time without disrupting training
- **Heterogeneous devices** — mix CPU and GPU volunteers freely
- **Gradient validation** — NaN/Inf gradients are detected and discarded
- **Heartbeat keepalive** — stale connections are detected and cleaned up
- **Cosine LR schedule** with warmup, matching nanoGPT defaults
- **Checkpoint saving** — compatible with nanoGPT checkpoint format
- **Compressed transfers** — model weights and gradients are zlib-compressed
