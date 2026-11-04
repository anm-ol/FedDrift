# Client drift per diffusion timestep

In federated training of a diffusion model, do clients disagree equally at every noise level, or more at some? This repo measures client gradient drift separately for each timestep band. It then tests a drift correction whose strength in each band follows that band's measured drift. The setup is DDPM + FedAvg on CIFAR-10 with one class per client.

---

## Main results

FID (10k samples, EMA model, seed 0) for 10 clients with one class each:

| Method | Round 25 | Round 50 | Round 100 |
|--------|:--------:|:--------:|:---------:|
| FedAvg | 134.8 | **52.0** | **43.35** |
| FedProx, uniform strength | 131.7 | 52.1 | 43.92 |
| **FedProx, drift-adaptive strength** | **112.4** | 53.3 | 43.84 |
| Distillation, uniform strength | 143.0 | 75.8 | 51.9 |
| **Distillation, drift-adaptive strength** | **125.4** | 74.8 | 57.8 |

- **Adaptive strength speeds up early training.** At round 25, adaptive FedProx is 22 FID ahead of FedAvg and 19 ahead of uniform FedProx. Adaptive distillation shows the same early lead. Since uniform strength gives no such gain, it comes from where the correction is placed.
- **The lead does not last.** By round 50 the FedProx variants are level with FedAvg, within the FedAvg seed-to-seed spread (43.35 vs 42.85 at round 100). Distillation ends worse than FedAvg.
- **Why:** early on, drift is largest at low noise (t < 400), which is also where it costs the most FID. After about round 20, drift moves to the middle band (t = 400–600), so a strength that follows drift size moves away from where it helps. Both corrections also pull clients toward the global model, so they can restrain a client but give it no new information.

### Reference: a small shared IID set, placed by timestep

This one is not data-free. All clients get the same 1000 random CIFAR-10 test images. Each local batch adds 32 of them, sampled only at timesteps in the chosen band. It shows which bands matter if some shared data were allowed.

| Shared images used at | Per batch | Round 100 (seed 0 / 1) | Round 150 (seed 0 / 1) |
|------|:--:|:--:|:--:|
| none (FedAvg) | – | 43.35 / 42.85 | 41.04 / – |
| all t | 32 | **27.10 / 26.34** | **27.86 / 27.33** |
| t < 400 | 13 | 29.17 / 30.83 | 29.40 / 30.78 |
| t < 400 | 32 | 41.84 / 43.54 | 41.76 / 42.39 |
| 400 ≤ t < 800 | 32 | 38.91 / 38.39 | 35.77 / 35.15 |

- **Even a small shared set closes most of the gap**: about 16 FID at round 100.
- **Most of that gain comes from the low band.** Under uniform sharing, about 13 of the 32 shared images per batch fall at t < 400. Giving only those 13 recovers most of the gain (29–31), while the same images used only in the middle band give much less (38–39).
- **Too much shared data in the low band hurts.** Putting all 32 shared images at t < 400 cancels the gain entirely.
- This agrees with the main results: low-noise timesteps are where client drift costs the most FID.

---

## How drift changes over training

Cross-client drift ζ² per timestep band, from FedAvg with one class per client (seed 0):

| Round | 0-200 | 200-400 | 400-600 | 600-800 | 800-1000 |
|-------|:-----:|:-------:|:-------:|:-------:|:--------:|
| 10 | **4.05e-3** | 2.77e-3 | 1.12e-3 | 2.14e-4 | 1.19e-5 |
| 25 | 1.35e-3 | 2.64e-3 | **3.83e-3** | 1.42e-3 | 1.04e-4 |
| 50 | 9.30e-4 | 2.71e-3 | **4.58e-3** | 2.38e-3 | 2.80e-4 |
| 100 | 6.33e-4 | 2.22e-3 | **4.19e-3** | 2.41e-3 | 2.77e-4 |

Drift is not uniform over timesteps, and its peak moves from low noise to mid noise within the first 25 rounds, then stays there.

---

## Methods

Both corrections are data-free: no client data leaves the client, and no shared dataset is needed.

- **FedProx (function space).** Client loss = ‖ε̂ − ε‖² + λ(t)·‖ε̂ − ε̂_global(x_t, t)‖², on the client's own noisy images.
- **Distillation.** Every 10 rounds the server samples 1000 images from the global model and sends them to the clients. Each client batch adds 32 of them, with only the term λ(t)·‖ε̂ − ε̂_global‖².
- **Strength per band.**
  - Uniform: λ̄ in every band.
  - Drift-adaptive: λ_b = λ̄ · ζ²_b / mean(ζ²), clipped to [0.2, 3] × λ̄, with ζ² re-measured on the global model every 10 rounds.
  - Both ramp up over 20 rounds. λ̄ = 0.4 for FedProx and 1.0 for distillation.

---

## Setup

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Install the torch build for your GPU first. CIFAR-10 downloads itself to `./data` on first use.

---

## Reproduce

```bash
python test_drift.py

bash run_fedavg.sh 0
bash run_prox.sh 0
bash run_distill.sh 0
bash run_shared.sh 0
```

The argument is the seed. Each script writes the client split if it is missing, trains for 100 rounds, saves checkpoints and drift measurements to `runs/`, and appends FIDs at rounds 25, 50 and 100 to `fid.txt`. `run_prox.sh` and `run_distill.sh` each run the uniform and adaptive versions side by side. `run_shared.sh` runs the four shared-data variants for 150 rounds, with FIDs at rounds 100 and 150. Each pair takes about 5 hours on one RTX PRO 6000.

---

## Files

| File | What it does |
|------|-------------|
| `data.py` | CIFAR-10 as one tensor, client splits |
| `make_partitions.py` | Writes the splits, prints the client x class counts |
| `model.py` | diffusers UNet2DModel, small (9M) and standard (35.7M) |
| `diffusion.py` | DDPM schedule, loss with explicit `t` and `eps`, sampler |
| `fed.py` | Local training with the per-band penalty, and FedAvg |
| `drift.py` | The per-band drift measurement |
| `measure_drift.py` | Federated training loop: drift measurement, adaptive strengths, probe images |
| `sample.py` | Sampling from a checkpoint |
| `fid.py` | FID against the CIFAR-10 training set |
| `test_drift.py` | Cats-only vs dogs-only clients must show more drift than mixed clients |
| `plot.py` | Drift per band, one figure per measured round |

---

## How drift is measured

For a fixed global model and one timestep band, every client computes its gradient on 256 of its own images. Drift is the average squared distance of the client gradients from their mean.

- all clients are measured at the same weights
- all clients get the same timesteps and the same noise, so only the images differ
- minibatch noise is removed, in two independent ways that are checked against each other

---

