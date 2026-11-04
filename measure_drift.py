import argparse
import json
import os
import time
from copy import deepcopy

import numpy as np
import torch
import wandb
from torchvision.utils import make_grid, save_image

from data import alpha_tag as tag
from data import load_cifar10, load_partition, parse_alpha, partition_path
from diffusion import Schedule
from drift import measure_drift
from fed import control_variate, fedavg, local_train
from model import UNet
from sample import generate


def to_wandb(x):
    grid = make_grid(x, nrow=8)
    return wandb.Image((grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=parse_alpha, default=0.1)
    parser.add_argument("--measure_alphas", type=parse_alpha, nargs="+", default=[0.1, 0.5, 5.0, None])
    parser.add_argument("--n_clients", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="small", choices=["small", "standard"])
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--ema_decay", type=float, default=0.8)
    parser.add_argument("--pool_range", type=int, nargs=2, default=None)
    parser.add_argument("--pool_size", type=int, default=0)
    parser.add_argument("--share_size", type=int, default=0)
    parser.add_argument("--share_range", type=int, nargs=2, default=[0, 1000])
    parser.add_argument("--share_batch", type=int, default=32)
    parser.add_argument("--share_weights", type=float, nargs="+", default=None)
    parser.add_argument("--share_adapt", action="store_true")
    parser.add_argument("--share_floor", type=float, default=0.02)
    parser.add_argument("--share_mix", type=float, default=0.0)
    parser.add_argument("--prox", type=float, default=0.0)
    parser.add_argument("--prox_range", type=int, nargs=2, default=None)
    parser.add_argument("--prox_adapt", action="store_true")
    parser.add_argument("--prox_ramp", type=int, default=0)
    parser.add_argument("--prox_clip", type=float, nargs=2, default=[0.2, 3.0])
    parser.add_argument("--prox_bands", type=int, default=5)
    parser.add_argument("--distill", type=float, default=0.0)
    parser.add_argument("--distill_n", type=int, default=1000)
    parser.add_argument("--distill_every", type=int, default=10)
    parser.add_argument("--distill_batch", type=int, default=32)
    parser.add_argument("--distill_start", type=int, default=0)
    parser.add_argument("--scaffold", action="store_true")
    parser.add_argument("--scaffold_range", type=int, nargs=2, default=None)
    parser.add_argument("--scaffold_scale", type=float, default=1.0)
    parser.add_argument("--cv_batches", type=int, default=8)
    parser.add_argument("--init", default=None)
    parser.add_argument("--measure_rounds", type=int, nargs="+", default=[0, 1, 5, 10, 25, 50])
    parser.add_argument("--measure_batch", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--partition_dir", default="./partitions")
    parser.add_argument("--out_dir", default="./runs")
    parser.add_argument("--name", default=None)
    parser.add_argument("--wandb_project", default="timestep-drift")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--resume_round", type=int, default=0)
    parser.add_argument("--resume_last", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    name = args.name or f"train{tag(args.alpha)}_E{args.local_epochs}_seed{args.seed}"
    if args.pool_range and not args.name:
        name += f"_pool{args.pool_range[0]}-{args.pool_range[1]}"
        if args.pool_size:
            name += f"_n{args.pool_size}"
    if args.share_size and not args.name:
        name += f"_share{args.share_range[0]}-{args.share_range[1]}_n{args.share_size}_b{args.share_batch}"
    if args.prox and not args.name:
        band = "all" if args.prox_range is None else f"{args.prox_range[0]}-{args.prox_range[1]}"
        name += f"_prox{args.prox:g}_{band}"
    if args.scaffold and not args.name:
        band = "all" if args.scaffold_range is None else f"{args.scaffold_range[0]}-{args.scaffold_range[1]}"
        name += f"_scaf{args.scaffold_scale:g}_{band}"
    resuming = args.resume_last or args.resume_round
    run_id = f"{args.name}_r" if (resuming and args.name) else args.name
    wandb.init(project=args.wandb_project, name=run_id or name, id=run_id, config=vars(args),
               mode="disabled" if args.no_wandb else "online", resume="allow" if resuming else None)

    x_all, _ = load_cifar10(args.data_root)
    x_all = x_all.to(device)
    sched = Schedule(device=device)

    def get_parts(alpha):
        path = partition_path(args.partition_dir, args.n_clients, alpha, args.seed)
        return [torch.from_numpy(p).to(device) for p in load_partition(path)]

    train_parts = get_parts(args.alpha)
    pool_idx = None
    if args.pool_size:
        pool_idx = torch.randperm(len(x_all), generator=torch.Generator().manual_seed(args.seed))[:args.pool_size].to(device)
    share = None
    if args.share_size:
        x_test, _ = load_cifar10(args.data_root, train=False)
        pick = torch.randperm(len(x_test), generator=torch.Generator().manual_seed(args.seed))[:args.share_size]
        share = x_test[pick].to(device)
        del x_test
    share_weights = None
    if args.share_weights:
        share_weights = torch.tensor(args.share_weights, device=device)
    measure_parts = {tag(a): get_parts(a) for a in args.measure_alphas}

    model = UNet(args.model).to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device))
    ema = deepcopy(model).eval().requires_grad_(False)
    print(f"{name} | {args.model} unet | {model.num_params() / 1e6:.1f}M params | {device}")

    base = os.path.join(args.out_dir, name)
    log = {"args": vars(args), "train_loss": {}, "drift": []}

    def measure(rnd):
        nonlocal share_weights, prox_rel
        t0 = time.time()
        scalars = {}
        for split, parts in measure_parts.items():
            res = measure_drift(model, x_all, parts, sched, batch=args.measure_batch,
                                repeats=args.repeats, seed=args.seed + rnd)
            if args.prox_adapt and rnd > 0 and split == tag(args.alpha) and len(res) == nb:
                z = torch.tensor([max(r["zeta2_cross"], 0.0) for r in res], device=device)
                if z.sum() > 0:
                    rel = (z / z.mean()).clamp(args.prox_clip[0], args.prox_clip[1])
                    prox_rel = rel / rel.mean()
                    log.setdefault("prox_rel", {})[rnd] = prox_rel.tolist()
                    base_lam = args.distill if args.distill else args.prox
                    print(f"  round {rnd} | prox strength per band from drift: " + "  ".join(f"{base_lam * v:.3f}" for v in prox_rel.tolist()))
            if args.share_adapt and split == tag(args.alpha) and sum(max(r["zeta2_cross"], 0.0) for r in res) > 0:
                z = torch.tensor([max(r["zeta2_cross"], 0.0) for r in res])
                w = torch.clamp(z / z.sum(), min=args.share_floor)
                w = w / w.sum()
                w = args.share_mix / len(w) + (1 - args.share_mix) * w
                share_weights = w.to(device)
                log.setdefault("share_weights", {})[rnd] = w.tolist()
                for b, r in enumerate(res):
                    scalars[f"share_weight/t{r['bucket'][0]}-{r['bucket'][1]}"] = w[b].item()
                print(f"  round {rnd} | share weights: " + "  ".join(f"{v:.3f}" for v in w.tolist()))
            for r in res:
                log["drift"].append({"round": rnd, "alpha": split, **r})
                lo, hi = r["bucket"]
                scalars[f"drift/{split}/t{lo}-{hi}"] = r["zeta2_cross"]
                scalars[f"drift_over_grad/{split}/t{lo}-{hi}"] = r["zeta2_cross_normalized"]
            row = "  ".join(f"{r['zeta2_cross']:9.2e}" for r in res)
            print(f"  round {rnd} | alpha {split:>4} | drift per bucket: {row}")
        with open(base + ".json", "w") as f:
            json.dump(log, f, indent=1)

        if rnd > 0:
            torch.save(model.state_dict(), f"{base}_round{rnd}.pt")
            torch.save(ema.state_dict(), f"{base}_round{rnd}_ema.pt")
            x = generate(model.eval(), args.n_samples, sched, device, args.seed)
            x_ema = generate(ema, args.n_samples, sched, device, args.seed)
            save_image(x, f"{base}_round{rnd}_samples.png", nrow=8, padding=2)
            save_image(x_ema, f"{base}_round{rnd}_ema_samples.png", nrow=8, padding=2)
            scalars["samples"] = to_wandb(x)
            scalars["samples_ema"] = to_wandb(x_ema)
        wandb.log(scalars, step=rnd)
        print(f"  measured in {time.time() - t0:.0f}s -> {base}.json")

    nb = args.prox_bands
    prox_rel = torch.ones(nb, device=device) if args.prox_adapt else None
    probes = None

    start_round = 1
    if args.resume_last:
        st = torch.load(base + "_last.pt", map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        ema.load_state_dict(st["ema"])
        log = st["log"]
        share_weights = st["share_weights"]
        if args.prox_adapt:
            prox_rel = st["prox_rel"].to(device)
        torch.set_rng_state(st["rng"].cpu())
        if device.type == "cuda" and "cuda_rng" in st:
            torch.cuda.set_rng_state_all([s.cpu() for s in st["cuda_rng"]])
        start_round = st["round"] + 1
    elif args.resume_round:
        r = args.resume_round
        model.load_state_dict(torch.load(f"{base}_round{r}.pt", map_location=device))
        ema.load_state_dict(torch.load(f"{base}_round{r}_ema.pt", map_location=device))
        with open(base + ".json") as f:
            log = json.load(f)
        torch.manual_seed(args.seed * 1000 + r)
        start_round = r + 1
    if start_round > 1:
        print(f"resumed at round {start_round}")

    if 0 in args.measure_rounds and start_round == 1:
        measure(0)

    for rnd in range(start_round, args.rounds + 1):
        t0 = time.time()
        states, sizes, losses, ratios, n_steps = [], [], [], [], 0
        corrs = [None] * len(train_parts)
        if args.scaffold:
            lo, hi = args.scaffold_range if args.scaffold_range is not None else (0, sched.T)
            frac = (hi - lo) / sched.T
            cs = [control_variate(model, x_all, idx, sched, args.scaffold_range, args.cv_batches, args.batch_size)
                  for idx in train_parts]
            w = [len(idx) / sum(len(i) for i in train_parts) for idx in train_parts]
            cbar = [sum(wk * c[j] for wk, c in zip(w, cs)) for j in range(len(cs[0]))]
            corrs = [[args.scaffold_scale * frac * (cb - ci) for cb, ci in zip(cbar, c)] for c in cs]
            del cs, cbar
        k = rnd - args.distill_start if args.distill else rnd
        ramp = 0.0 if k <= 0 else (min(1.0, k / args.prox_ramp) if args.prox_ramp else 1.0)
        lam = (args.distill if args.distill else args.prox) * ramp
        if args.distill and k > 0 and (probes is None or (k - 1) % args.distill_every == 0):
            tg = time.time()
            probes = generate(ema, args.distill_n, sched, device, seed=args.seed * 7919 + rnd, batch=250) * 2 - 1
            grid = (probes[:64] + 1) / 2
            save_image(grid, f"{base}_probes_round{rnd}.png", nrow=8, padding=2)
            wandb.log({"probes": to_wandb(grid)}, step=rnd)
            print(f"  round {rnd} | generated {len(probes)} probe images from the global model in {time.time() - tg:.0f}s")
        devs = []
        for idx, corr in zip(train_parts, corrs):
            local = deepcopy(model)
            ds = [torch.zeros(nb, device=device), torch.zeros(nb, device=device)] if (args.prox or args.distill) else None
            loss, steps, ratio = local_train(local, x_all, idx, sched, args.local_epochs, args.batch_size, args.lr,
                                             pool_range=args.pool_range, ref=model,
                                             prox=0.0 if args.distill else lam,
                                             prox_range=args.prox_range, corr=corr, pool_idx=pool_idx,
                                             share=share, share_range=args.share_range,
                                             share_batch=args.share_batch, share_weights=share_weights,
                                             prox_rel=prox_rel, dev_stats=ds, probes=probes,
                                             distill=lam if args.distill else 0.0, distill_batch=args.distill_batch)
            if ds is not None:
                devs.append(ds[0] / ds[1].clamp(min=1))
            ratios.append(ratio)
            states.append(local.state_dict())
            sizes.append(len(idx))
            losses.append(loss)
            n_steps += steps
        start = {k: v.float() for k, v in model.state_dict().items()}
        ups = [torch.cat([(st[k].float() - start[k]).flatten() for k in start]) for st in states]
        ubar = sum(ups) / len(ups)
        spread = (sum((u - ubar).pow(2).sum() for u in ups) / sum(u.pow(2).sum() for u in ups)).item()
        del start, ups, ubar
        model.load_state_dict(fedavg(states, sizes))

        if rnd == 1:
            ema.load_state_dict(model.state_dict())
        else:
            for e, p in zip(ema.parameters(), model.parameters()):
                e.data.lerp_(p.data, 1 - args.ema_decay)

        loss = sum(l * n for l, n in zip(losses, sizes)) / sum(sizes)
        dt = time.time() - t0
        log["train_loss"][rnd] = loss
        log.setdefault("update_spread", {})[rnd] = spread
        corr_ratio = sum(ratios) / len(ratios)
        scal = {"loss": loss, "it_per_s": n_steps / dt, "corr_over_grad": corr_ratio, "update_spread": spread}
        extra = ""
        if devs:
            dev = sum(n * d for n, d in zip(sizes, devs)) / sum(sizes)
            used = prox_rel.clone() if prox_rel is not None else torch.ones(nb, device=device)
            log.setdefault("prox_dev", {})[rnd] = dev.tolist()
            log.setdefault("prox_strength", {})[rnd] = (lam * used).tolist()
            scal["prox_lambda_mean"] = lam
            for b in range(nb):
                lo, hi = b * sched.T // nb, (b + 1) * sched.T // nb
                scal[f"prox_dev/t{lo}-{hi}"] = dev[b].item()
                scal[f"prox_strength/t{lo}-{hi}"] = (lam * used[b]).item()
            extra = " | dev " + " ".join(f"{v:.2e}" for v in dev.tolist()) + " | strength " + " ".join(f"{lam * v:.3f}" for v in used.tolist())
        wandb.log(scal, step=rnd)
        print(f"round {rnd}/{args.rounds} | loss {loss:.4f} | corr/grad {corr_ratio:.3f} | update spread {spread:.3f} | {n_steps} steps in {dt:.0f}s ({n_steps / dt:.1f} it/s){extra}")

        if rnd in args.measure_rounds:
            measure(rnd)

        state = {"model": model.state_dict(), "ema": ema.state_dict(), "round": rnd, "log": log,
                 "share_weights": share_weights, "rng": torch.get_rng_state(),
                 "prox_rel": prox_rel}
        if device.type == "cuda":
            state["cuda_rng"] = torch.cuda.get_rng_state_all()
        torch.save(state, base + "_last.tmp")
        os.replace(base + "_last.tmp", base + "_last.pt")

    wandb.finish()


if __name__ == "__main__":
    main()
