import argparse
import json
import math
import os
import time
from copy import deepcopy

import numpy as np
import torch
import wandb
from torchvision.utils import make_grid, save_image

from data import alpha_tag as tag
from data import DATASETS, IMG_SIZE, load_cifar100, load_dataset, load_partition, parse_alpha, partition_path
from diffusion import Schedule, ddpm_loss, q_sample, target
from drift import measure_drift
from fed import control_variate, fedavg, local_train
from model import UNet
from sample import generate


def to_wandb(x):
    grid = make_grid(x, nrow=8)
    return wandb.Image((grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", choices=DATASETS)
    parser.add_argument("--objective", default="ddpm", choices=["ddpm", "fm"])
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
    parser.add_argument("--share_source", default="cifar10_test", choices=["cifar10_test", "cifar100"])
    parser.add_argument("--share_range", type=int, nargs=2, default=[0, 1000])
    parser.add_argument("--share_batch", type=int, default=32)
    parser.add_argument("--share_weights", type=float, nargs="+", default=None)
    parser.add_argument("--share_adapt", action="store_true")
    parser.add_argument("--share_floor", type=float, default=0.02)
    parser.add_argument("--share_mix", type=float, default=0.0)
    parser.add_argument("--mix", action="store_true")
    parser.add_argument("--mix_n", type=int, default=1000)
    parser.add_argument("--mix_every", type=int, default=10)
    parser.add_argument("--mix_start", type=int, default=10)
    parser.add_argument("--mix_ramp", type=int, default=10)
    parser.add_argument("--mix_steps", type=int, default=100)
    parser.add_argument("--mix_batch", type=int, default=128)
    parser.add_argument("--mix_lr", type=float, default=1e-4)
    parser.add_argument("--mix_draws", type=int, default=8)
    parser.add_argument("--repair", default="none", choices=["none", "uniform", "spread", "low"])
    parser.add_argument("--repair_n", type=int, default=1000)
    parser.add_argument("--repair_held", type=int, default=500)
    parser.add_argument("--repair_steps", type=int, default=20)
    parser.add_argument("--repair_batch", type=int, default=128)
    parser.add_argument("--repair_lr", type=float, default=2e-4)
    parser.add_argument("--repair_low", type=int, default=400)
    parser.add_argument("--regmean", default="none", choices=["none", "unif", "low"])
    parser.add_argument("--regmean_a", type=float, default=0.5)
    parser.add_argument("--regmean_n", type=int, default=1024)
    parser.add_argument("--regmean_low", type=int, default=400)
    parser.add_argument("--regmean_rel", type=float, default=1e-4)
    parser.add_argument("--regmean_omega", type=float, nargs=5, default=None)
    parser.add_argument("--regmean_rho", default="unif", choices=["unif", "gen"])
    parser.add_argument("--regmean_strength", type=float, default=1.0)
    parser.add_argument("--regmean_warmup", type=int, default=10)
    parser.add_argument("--regmean_every", type=int, default=10)
    parser.add_argument("--regmean_tau", type=float, default=0.05)
    parser.add_argument("--regmean_clip", type=float, nargs=2, default=[0.2, 2.0])
    parser.add_argument("--regmean_val", type=int, default=500)
    parser.add_argument("--heldout_n", type=int, default=1000)
    parser.add_argument("--sample_every", type=int, default=0)
    parser.add_argument("--ref_norm", default="")
    parser.add_argument("--share_value", action="store_true")
    parser.add_argument("--value_probe", type=int, default=200)
    parser.add_argument("--value_n", type=int, default=128)
    parser.add_argument("--value_lr", type=float, default=0.5)
    parser.add_argument("--value_ema", type=float, default=0.5)
    parser.add_argument("--value_floor", type=float, default=0.05)
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
               mode="disabled" if args.no_wandb else os.environ.get("WANDB_MODE", "online"), resume="allow" if resuming else None)

    x_all, y_all = load_dataset(args.dataset, args.data_root)
    x_all = x_all.to(device)
    sched = Schedule(device=device, kind=args.objective)
    if args.objective != "ddpm" and args.mix:
        raise ValueError("--mix blends eps predictions with alphabar; DDPM only")
    if args.share_source == "cifar100" and args.share_size and args.dataset != "cifar10":
        raise ValueError("--share_source cifar100 is only meaningful for cifar10")

    def get_parts(alpha):
        path = partition_path(args.partition_dir, args.n_clients, alpha, args.seed, args.dataset)
        return [torch.from_numpy(p).to(device) for p in load_partition(path)]

    train_parts = get_parts(args.alpha)
    pool_idx = None
    if args.pool_size:
        pool_idx = torch.randperm(len(x_all), generator=torch.Generator().manual_seed(args.seed))[:args.pool_size].to(device)
    share = None
    if args.share_size:
        x_test, _ = load_cifar100(args.data_root) if args.share_source == "cifar100" else load_dataset(args.dataset, args.data_root, train=False)
        pick = torch.randperm(len(x_test), generator=torch.Generator().manual_seed(args.seed))[:args.share_size]
        share = x_test[pick].to(device)
        del x_test
    pub = pub_held = None
    if args.repair != "none":
        x_test, _ = load_dataset(args.dataset, args.data_root, train=False)
        pick = torch.randperm(len(x_test), generator=torch.Generator().manual_seed(args.seed))
        pub = x_test[pick[:args.repair_n]].to(device)
        pub_held = x_test[pick[args.repair_n:args.repair_n + args.repair_held]].to(device)
        del x_test
    held = None
    if args.share_value:
        held = share[len(share) - args.value_probe:]
        share = share[:len(share) - args.value_probe]
    share_weights = None
    if args.share_weights:
        share_weights = torch.tensor(args.share_weights, device=device)
    measure_parts = {tag(a): get_parts(a) for a in args.measure_alphas}

    model = UNet(args.model, IMG_SIZE[args.dataset]).to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device))
    ema = deepcopy(model).eval().requires_grad_(False)
    print(f"{name} | {args.dataset} | {args.objective} | {args.model} unet | {model.num_params() / 1e6:.1f}M params | {device}")

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
    vbar = None
    if args.share_value:
        share_weights = torch.full((nb,), 1.0 / nb, device=device)

    def draws(rnd):
        g = torch.Generator(device=device).manual_seed(args.seed * 100003 + rnd)
        width = sched.T // nb
        out = []
        for b in range(nb):
            i = torch.randint(0, len(share), (args.value_n,), device=device, generator=g)
            t = b * width + torch.randint(0, width, (args.value_n,), device=device, generator=g)
            out.append((share[i], t, torch.randn(share[i].shape, device=device, generator=g)))
        t = torch.randint(0, sched.T, (len(held),), device=device, generator=g)
        out.append((held, t, torch.randn(held.shape, device=device, generator=g)))
        return out

    def grad_vec(net, x0, t, eps):
        params = [p for p in net.parameters() if p.requires_grad]
        g = torch.autograd.grad(ddpm_loss(net, x0, t, eps, sched), params)
        return torch.cat([v.flatten() for v in g])

    mix_probes = None
    abar = sched.sqrt_ab.pow(2)

    @torch.no_grad()
    def route(nets, x0, rnd):
        g = torch.Generator(device=device).manual_seed(args.seed * 31337 + rnd)
        errs = []
        for i in range(0, len(x0), 250):
            xb = x0[i:i + 250]
            err = torch.zeros(len(nets), len(xb), device=device)
            for _ in range(args.mix_draws):
                t = torch.randint(0, sched.T, (len(xb),), device=device, generator=g)
                eps = torch.randn(xb.shape, device=device, generator=g)
                xt = q_sample(xb, t, eps, sched)
                with torch.autocast(device.type, enabled=device.type == "cuda"):
                    e = torch.stack([(n(xt, t).float() - target(xb, eps, sched)).pow(2).mean(dim=(1, 2, 3)) for n in nets])
                err += e / e.mean(0, keepdim=True)
            errs.append(err)
        err = torch.cat(errs, dim=1)
        return (err - err.mean(1, keepdim=True)).argmin(0)

    def mix_distill(nets, x0, k, scale, rnd):
        opt = torch.optim.Adam(model.parameters(), lr=args.mix_lr * scale)
        g = torch.Generator(device=device).manual_seed(args.seed * 7777 + rnd)
        gap = torch.zeros(nb, device=device)
        cnt = torch.zeros(nb, device=device)
        model.train()
        for s in range(args.mix_steps):
            i = torch.randint(0, len(x0), (args.mix_batch,), device=device, generator=g)
            t = torch.randint(0, sched.T, (args.mix_batch,), device=device, generator=g)
            eps = torch.randn(x0[i].shape, device=device, generator=g)
            xt = q_sample(x0[i], t, eps, sched)
            with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
                outs = torch.stack([n(xt, t).float() for n in nets])
            match = outs[k[i], torch.arange(args.mix_batch, device=device)]
            a = abar[t].view(-1, 1, 1, 1)
            target = a * match + (1 - a) * outs.mean(0)
            pred = model(xt, t).float()
            d = (pred - target).pow(2).mean(dim=(1, 2, 3))
            if s == 0:
                b = (t * nb) // sched.T
                gap.index_add_(0, b, d.detach())
                cnt.index_add_(0, b, torch.ones_like(d))
            opt.zero_grad(set_to_none=True)
            d.mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        return gap / cnt.clamp(min=1)

    ropt = torch.optim.Adam(model.parameters(), lr=args.repair_lr) if args.repair != "none" else None

    @torch.no_grad()
    def client_spread(nets, xt, t):
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            outs = torch.stack([n(xt, t).float() for n in nets])
        return (outs - outs.mean(0)).pow(2).mean(dim=(2, 3, 4)).mean(0)

    def repair(nets, rnd):
        g = torch.Generator(device=device).manual_seed(args.seed * 9973 + rnd)
        sb = torch.zeros(nb, device=device)
        wb = torch.zeros(nb, device=device)
        cb = torch.zeros(nb, device=device)
        model.train()
        for s in range(args.repair_steps):
            i = torch.randint(0, len(pub), (args.repair_batch,), device=device, generator=g)
            t = torch.randint(0, sched.T, (args.repair_batch,), device=device, generator=g)
            if args.repair == "low":
                t = t % args.repair_low
            eps = torch.randn(pub[i].shape, device=device, generator=g)
            xt = q_sample(pub[i], t, eps, sched)
            w = torch.ones(args.repair_batch, device=device)
            if args.repair == "spread" or s == 0:
                sp = client_spread(nets, xt, t)
                if args.repair == "spread":
                    w = sp / sp.mean().clamp(min=1e-12)
                b = (t * nb) // sched.T
                sb.index_add_(0, b, sp)
                wb.index_add_(0, b, w)
                cb.index_add_(0, b, torch.ones_like(w))
            d = (model(xt, t).float() - target(pub[i], eps, sched)).pow(2).mean(dim=(1, 2, 3))
            ropt.zero_grad(set_to_none=True)
            (w * d).mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            ropt.step()
        return sb / cb.clamp(min=1), wb / cb.clamp(min=1)

    @torch.no_grad()
    def low_t_loss(x0):
        g = torch.Generator(device=device).manual_seed(args.seed * 7 + 1)
        t = torch.randint(0, args.repair_low, (len(x0),), device=device, generator=g)
        eps = torch.randn(x0.shape, device=device, generator=g)
        model.eval()
        return (model(q_sample(x0, t, eps, sched), t).float() - target(x0, eps, sched)).pow(2).mean().item()

    def values(net, d):
        gh = grad_vec(net, *d[-1])
        return torch.stack([torch.dot(grad_vec(net, *d[b]), gh) for b in range(nb)])

    @torch.no_grad()
    def band_loss_of(net, x, seed):
        net.eval()
        g = torch.Generator(device=device).manual_seed(seed)
        out = []
        for b in range(nb):
            lo, hi = b * sched.T // nb, (b + 1) * sched.T // nb
            t = torch.randint(lo, hi, (len(x),), device=device, generator=g)
            eps = torch.randn(x.shape, device=device, generator=g)
            tot = 0.0
            for i in range(0, len(x), 250):
                o = net(q_sample(x[i:i + 250], t[i:i + 250], eps[i:i + 250], sched), t[i:i + 250]).float()
                tot += (o - target(x[i:i + 250], eps[i:i + 250], sched)).pow(2).mean(dim=(1, 2, 3)).sum().item()
            out.append(tot / len(x))
        return out

    rg = None
    if args.regmean == "unif" and not args.regmean_omega:
        x_te, y_te = load_dataset(args.dataset, args.data_root, train=False)
        gsel = torch.Generator().manual_seed(args.seed + 4242)
        nc = int(max(y_all.max(), y_te.max())) + 1
        te_count = np.bincount(y_te, minlength=nc).astype(np.float64)
        val = []
        for idx in train_parts:
            # test images drawn to match this client's label mix (= its own class under the
            # one-class-per-client CIFAR split; a mixture for Tiny-ImageNet/CelebA splits)
            mix = np.bincount(y_all[idx.cpu().numpy()], minlength=nc).astype(np.float64)
            w = torch.from_numpy(mix[y_te] / np.maximum(te_count[y_te], 1))
            ids = torch.multinomial(w, min(args.regmean_val, int((w > 0).sum())), replacement=False, generator=gsel).numpy()
            val.append(x_te[np.sort(ids)].to(device))
        hold = x_te[torch.randperm(len(x_te), generator=gsel)[:args.heldout_n]].to(device)
        del x_te
        rg = {"val": val, "hold": hold, "rho": [args.regmean_strength] * nb, "signal": None}

    start_round = 1
    if args.resume_last:
        st = torch.load(base + "_last.pt", map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        ema.load_state_dict(st["ema"])
        log = st["log"]
        share_weights = st["share_weights"]
        if rg is not None and st.get("regmean_rg"):
            rg.update(st["regmean_rg"])
        vbar = st.get("value_bar")
        if ropt is not None and st.get("repair_opt") is not None:
            ropt.load_state_dict(st["repair_opt"])
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
        reps = []
        if args.share_value:
            d = draws(rnd)
            vg = values(model, d)
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
            if args.share_value:
                reps.append(values(local, d) - vg)
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
        prev = {k: v.clone() for k, v in model.state_dict().items()} if args.regmean != "none" else None
        model.load_state_dict(fedavg(states, sizes))

        reg_info = ""
        if prev is not None and rg is not None:
            from regmean import grams
            tg = time.time()
            rb = [(b * sched.T // nb, (b + 1) * sched.T // nb) for b in range(nb)]
            gg = torch.Generator(device=device).manual_seed(args.seed * 7717 + rnd)
            due = args.regmean_rho == "gen" and rnd > args.regmean_warmup and (rnd - args.regmean_warmup - 1) % args.regmean_every == 0
            net = deepcopy(model)
            if due:
                net.load_state_dict(prev)
                l0 = [band_loss_of(net, v, args.seed * 31 + k) for k, v in enumerate(rg["val"])]
                gain = [0.0] * nb
            G = []
            for k, (st, idx) in enumerate(zip(states, train_parts)):
                net.load_state_dict(st)
                G.append(grams(net, x_all[idx], sched, rb, args.regmean_n, 256, gg))
                if due:
                    lk = band_loss_of(net, rg["val"][k], args.seed * 31 + k)
                    for b in range(nb):
                        gain[b] += (l0[k][b] - lk[b]) / l0[k][b] / len(states)
            del net
            if due:
                raw = [max(args.regmean_clip[0], math.exp(min(gb / args.regmean_tau, 50.0))) for gb in gain]
                clip = lambda c: [min(args.regmean_clip[1], max(args.regmean_clip[0], c * v)) for v in raw]
                lo_c, hi_c = -60.0, 60.0
                for _ in range(200):
                    mid = (lo_c + hi_c) / 2
                    lo_c, hi_c = (mid, hi_c) if sum(clip(math.exp(mid))) < nb else (lo_c, mid)
                shp = clip(math.exp((lo_c + hi_c) / 2))
                m = sum(shp) / nb
                rg["rho"] = [args.regmean_strength * v / m for v in shp]
                rg["signal"] = gain
            rho = rg["rho"]
            a = args.regmean_a
            pi = [n_ / sum(sizes) for n_ in sizes]
            sd = model.state_dict()
            nf = nr = 0.0
            worst = 0.0
            for name in G[0]:
                if "time" in name:
                    continue
                key = name + ".weight"
                ds = [(st[key].float() - prev[key].float()).movedim(1, -1) for st in states]
                gs = [[a * g[name][b] + (1 - a) * torch.diag(torch.diagonal(g[name][b])) for b in range(nb)] for g in G]
                bar = [sum(p * gk[b] for p, gk in zip(pi, gs)) for b in range(nb)]
                S = sum(bar)
                Sr = S + args.regmean_rel * S.diagonal().mean() * torch.eye(len(S), device=S.device)
                dbar = sum(p * d for p, d in zip(pi, ds))
                N = dbar @ S
                for b in range(nb):
                    N = N + rho[b] * (sum(p * (d @ gk[b]) for p, d, gk in zip(pi, ds, gs)) - dbar @ bar[b])
                u = torch.linalg.solve(Sr, N.reshape(-1, Sr.shape[0]).T).T.reshape(N.shape).movedim(-1, 1)
                fa_ = dbar.movedim(-1, 1)
                nf += fa_.pow(2).sum().item()
                nr += u.pow(2).sum().item()
                worst = max(worst, (u.norm() / (fa_.norm() + 1e-12)).item())
                sd[key] = (prev[key].float() + u).to(prev[key].dtype)
            model.load_state_dict(sd)
            del G, prev, sd
            wn = sum(p.detach().float().pow(2).sum().item() for p in model.parameters()) ** 0.5
            hl = band_loss_of(model, rg["hold"], args.seed * 53)
            rec = {"u": nr ** 0.5, "fedavg_u": nf ** 0.5, "max_layer_ratio": worst, "weight_norm": wn, "rho": list(rho), "signal": rg["signal"], "heldout": hl}
            ref_path = args.ref_norm.format(rnd) if args.ref_norm else ""
            if ref_path and os.path.exists(ref_path):
                ref_sd = torch.load(ref_path, map_location=device)
                rn = sum(ref_sd[n_].float().pow(2).sum().item() for n_, _ in model.named_parameters()) ** 0.5
                rec["weight_norm_vs_ref"] = wn / rn
                del ref_sd
            log.setdefault("regmean", {})[rnd] = rec
            trip = ""
            if "weight_norm_vs_ref" in rec:
                trip = f" | |w| vs FedAvg {rec['weight_norm_vs_ref']:.3f}" + (" TRIPWIRE" if rec["weight_norm_vs_ref"] > 1.15 else "")
            sig = (" | signal % " + " ".join(f"{100 * v:+.2f}" for v in rg["signal"])) if due else ""
            reg_info = (f" | regmean |u| {nr ** 0.5:.3f} vs avg {nf ** 0.5:.3f} | max layer {worst:.1f} | |w| {wn:.2f} | rho " + " ".join(f"{v:.2f}" for v in rho)
                        + sig + " | heldout " + " ".join(f"{v:.5f}" for v in hl) + trip + f" | {time.time() - tg:.0f}s")
        elif prev is not None:
            from regmean import grams, regmean
            tg = time.time()
            om = args.regmean_omega or [1.0 if (args.regmean == "unif" or (b + 1) * sched.T // nb <= args.regmean_low) else 0.0 for b in range(nb)]
            keep = [b for b in range(nb) if om[b] > 0]
            wk = torch.tensor([om[b] for b in keep], device=device).view(-1, 1, 1)
            rb = [(b * sched.T // nb, (b + 1) * sched.T // nb) for b in keep]
            gg = torch.Generator(device=device).manual_seed(args.seed * 7717 + rnd)
            net = deepcopy(model)
            G = []
            for st, idx in zip(states, train_parts):
                net.load_state_dict(st)
                G.append({k: (wk * v).sum(0) for k, v in grams(net, x_all[idx], sched, rb, args.regmean_n, 256, gg).items()})
            del net
            pi = [s / sum(sizes) for s in sizes]
            sd = model.state_dict()
            nf = nr = 0.0
            for name in G[0]:
                if "time" in name:
                    continue
                key = name + ".weight"
                ds = [st[key].float() - prev[key].float() for st in states]
                u = regmean(ds, [g[name] for g in G], pi, args.regmean_a, args.regmean_rel)
                nf += sum(p * d for p, d in zip(pi, ds)).pow(2).sum().item()
                nr += u.pow(2).sum().item()
                sd[key] = (prev[key].float() + u).to(prev[key].dtype)
            model.load_state_dict(sd)
            del G, prev, sd
            log.setdefault("regmean_norm", {})[rnd] = [nr ** 0.5, nf ** 0.5]
            reg_info = f" | regmean |u| {nr ** 0.5:.3f} vs fedavg {nf ** 0.5:.3f} | {time.time() - tg:.0f}s"

        mix_info = ""
        mk = rnd - args.mix_start
        if args.mix and mk > 0:
            tm = time.time()
            if mix_probes is None or (mk - 1) % args.mix_every == 0:
                mix_probes = generate(ema, args.mix_n, sched, device, seed=args.seed * 4243 + rnd, batch=250) * 2 - 1
                save_image((mix_probes[:64] + 1) / 2, f"{base}_mixprobes_round{rnd}.png", nrow=8, padding=2)
            nets = []
            for st in states:
                n = deepcopy(model)
                n.load_state_dict(st)
                nets.append(n.eval().requires_grad_(False))
            k = route(nets, mix_probes, rnd)
            scale = min(1.0, mk / args.mix_ramp) if args.mix_ramp else 1.0
            gap = mix_distill(nets, mix_probes, k, scale, rnd)
            del nets
            counts = torch.bincount(k, minlength=len(states)).tolist()
            log.setdefault("mix_gap", {})[rnd] = gap.tolist()
            log.setdefault("mix_counts", {})[rnd] = counts
            mix_info = f" | mix gap " + " ".join(f"{v:.2e}" for v in gap.tolist()) + f" | routed {counts} | lr x{scale:.2f} | {time.time() - tm:.0f}s"

        rep_info = ""
        if ropt is not None:
            tr = time.time()
            nets = []
            for st in states:
                n = deepcopy(model)
                n.load_state_dict(st)
                nets.append(n.eval().requires_grad_(False))
            sp, wt = repair(nets, rnd)
            del nets
            lt, lh = low_t_loss(pub[:args.repair_held]), low_t_loss(pub_held)
            log.setdefault("repair_spread", {})[rnd] = sp.tolist()
            log.setdefault("repair_weight", {})[rnd] = wt.tolist()
            log.setdefault("repair_lowt", {})[rnd] = [lt, lh]
            rep_info = (" | spread " + " ".join(f"{v:.2e}" for v in sp.tolist()) + " | weight " + " ".join(f"{v:.2f}" for v in wt.tolist())
                        + f" | low-t loss repair {lt:.4f} held {lh:.4f} | {time.time() - tr:.0f}s")

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
        extra = mix_info + rep_info + reg_info
        if reg_info and rnd in log.get("regmean", {}):
            r_ = log["regmean"][rnd]
            scal["regmean/update_norm"], scal["regmean/fedavg_update_norm"] = r_["u"], r_["fedavg_u"]
            scal["regmean/weight_norm"], scal["regmean/max_layer_ratio"] = r_["weight_norm"], r_["max_layer_ratio"]
            if "weight_norm_vs_ref" in r_:
                scal["regmean/weight_norm_vs_fedavg"] = r_["weight_norm_vs_ref"]
            for b in range(nb):
                lo, hi = b * sched.T // nb, (b + 1) * sched.T // nb
                scal[f"regmean/rho/t{lo}-{hi}"] = r_["rho"][b]
                scal[f"heldout_loss/t{lo}-{hi}"] = r_["heldout"][b]
                if r_["signal"] is not None:
                    scal[f"regmean/signal/t{lo}-{hi}"] = r_["signal"][b]
        elif reg_info:
            scal["regmean_norm"], scal["fedavg_norm"] = log["regmean_norm"][rnd]
        if args.sample_every and rnd % args.sample_every == 0 and rnd not in args.measure_rounds:
            xs = generate(ema, 64, sched, device, seed=args.seed, batch=64)
            save_image(xs, f"{base}_round{rnd}_ema_samples.png", nrow=8, padding=2)
            scal["samples_ema"] = to_wandb(xs)
        if rep_info:
            for b in range(nb):
                lo, hi = b * sched.T // nb, (b + 1) * sched.T // nb
                scal[f"repair_spread/t{lo}-{hi}"] = log["repair_spread"][rnd][b]
                scal[f"repair_weight/t{lo}-{hi}"] = log["repair_weight"][rnd][b]
            scal["lowt_loss/repair_set"], scal["lowt_loss/held_out"] = log["repair_lowt"][rnd]
        if mix_info:
            for b, v in enumerate(log["mix_gap"][rnd]):
                scal[f"mix_gap/t{b * sched.T // nb}-{(b + 1) * sched.T // nb}"] = v
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
        if reps:
            r = sum(n * v for n, v in zip(sizes, reps)) / sum(sizes)
            vbar = r if vbar is None else args.value_ema * vbar + (1 - args.value_ema) * r
            used = share_weights.clone()
            w = share_weights * torch.exp(args.value_lr * vbar / vbar.abs().max().clamp(min=1e-20))
            w = w / w.sum()
            share_weights = (1 - args.value_floor) * w + args.value_floor / nb
            log.setdefault("value_repair", {})[rnd] = r.tolist()
            log.setdefault("share_weights", {})[rnd] = used.tolist()
            for b in range(nb):
                lo, hi = b * sched.T // nb, (b + 1) * sched.T // nb
                scal[f"value_repair/t{lo}-{hi}"] = r[b].item()
                scal[f"share_weight/t{lo}-{hi}"] = used[b].item()
            extra += " | repair " + " ".join(f"{v:.2e}" for v in r.tolist()) + " | weights used " + " ".join(f"{v:.3f}" for v in used.tolist())
        wandb.log(scal, step=rnd)
        print(f"round {rnd}/{args.rounds} | loss {loss:.4f} | corr/grad {corr_ratio:.3f} | update spread {spread:.3f} | {n_steps} steps in {dt:.0f}s ({n_steps / dt:.1f} it/s){extra}")

        if rnd in args.measure_rounds:
            measure(rnd)

        state = {"model": model.state_dict(), "ema": ema.state_dict(), "round": rnd, "log": log,
                 "share_weights": share_weights, "rng": torch.get_rng_state(),
                 "prox_rel": prox_rel, "value_bar": vbar,
                 "repair_opt": ropt.state_dict() if ropt is not None else None,
                 "regmean_rg": {"rho": rg["rho"], "signal": rg["signal"]} if rg is not None else None}
        if device.type == "cuda":
            state["cuda_rng"] = torch.cuda.get_rng_state_all()
        torch.save(state, base + "_last.tmp")
        os.replace(base + "_last.tmp", base + "_last.pt")

    wandb.finish()


if __name__ == "__main__":
    main()
