import torch
import torch.nn as nn

from data import random_flip
from diffusion import ddpm_loss, q_sample


def local_train(model, x_all, idx, sched, epochs, batch_size, lr, grad_clip=1.0, pool_range=None,
                ref=None, prox=0.0, prox_range=None, corr=None, pool_idx=None, share=None, share_range=None,
                share_batch=0, share_weights=None, prox_rel=None, dev_stats=None, probes=None, distill=0.0,
                distill_batch=32):
    device = x_all.device
    params = list(model.parameters())
    gsum = torch.zeros((), device=device)
    use_amp = device.type == "cuda"
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    total, steps = 0.0, 0
    for _ in range(epochs):
        perm = idx[torch.randperm(len(idx), device=device)]
        for i in range(0, len(perm), batch_size):
            x0 = random_flip(x_all[perm[i:i + batch_size]])
            t = torch.randint(0, sched.T, (x0.shape[0],), device=device)
            eps = torch.randn_like(x0)
            if pool_range is not None:
                m = (t >= pool_range[0]) & (t < pool_range[1])
                pooled = torch.randint(0, len(x_all), (int(m.sum()),), device=device)
                if pool_idx is not None:
                    pooled = pool_idx[torch.randint(0, len(pool_idx), (int(m.sum()),), device=device)]
                x0[m] = random_flip(x_all[pooled])
            if share is not None:
                xs = random_flip(share[torch.randint(0, len(share), (share_batch,), device=device)])
                if share_weights is None:
                    ts = torch.randint(share_range[0], share_range[1], (share_batch,), device=device)
                else:
                    width = sched.T // len(share_weights)
                    band = torch.multinomial(share_weights, share_batch, replacement=True)
                    ts = band * width + torch.randint(0, width, (share_batch,), device=device)
                x0 = torch.cat([x0, xs])
                t = torch.cat([t, ts])
                eps = torch.cat([eps, torch.randn_like(xs)])

            nd = distill_batch if (probes is not None and distill > 0 and ref is not None) else 0
            if nd:
                xp = random_flip(probes[torch.randint(0, len(probes), (nd,), device=device)])
                tp = torch.randint(0, sched.T, (nd,), device=device)
                ep = torch.randn_like(xp)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=use_amp):
                if nd:
                    n = x0.shape[0]
                    xt = q_sample(torch.cat([x0, xp]), torch.cat([t, tp]), torch.cat([eps, ep]), sched)
                    pred = model(xt, torch.cat([t, tp]))
                    with torch.no_grad():
                        anchor = ref(xt[n:], tp)
                    w = torch.full_like(tp, distill, dtype=torch.float)
                    if prox_rel is not None:
                        w = w * prox_rel[(tp * len(prox_rel)) // sched.T]
                    loss = (pred[:n] - eps).pow(2).mean()
                    full = loss + (w.view(-1, 1, 1, 1) * (pred[n:] - anchor).pow(2)).mean()
                    if dev_stats is not None:
                        with torch.no_grad():
                            d = (pred[n:].float() - anchor.float()).pow(2).mean(dim=(1, 2, 3))
                            b = (tp * len(dev_stats[0])) // sched.T
                            dev_stats[0].index_add_(0, b, d)
                            dev_stats[1].index_add_(0, b, torch.ones_like(d))
                elif ref is None or prox == 0:
                    loss = ddpm_loss(model, x0, t, eps, sched)
                    full = loss
                else:
                    xt = q_sample(x0, t, eps, sched)
                    pred = model(xt, t)
                    with torch.no_grad():
                        anchor = ref(xt, t)
                    w = torch.full_like(t, prox, dtype=torch.float)
                    if prox_range is not None:
                        w = w * ((t >= prox_range[0]) & (t < prox_range[1])).float()
                    if prox_rel is not None:
                        w = w * prox_rel[(t * len(prox_rel)) // sched.T]
                    loss = (pred - eps).pow(2).mean()
                    full = loss + (w.view(-1, 1, 1, 1) * (pred - anchor).pow(2)).mean()
                    if dev_stats is not None:
                        with torch.no_grad():
                            d = (pred.float() - anchor.float()).pow(2).mean(dim=(1, 2, 3))
                            b = (t * len(dev_stats[0])) // sched.T
                            dev_stats[0].index_add_(0, b, d)
                            dev_stats[1].index_add_(0, b, torch.ones_like(d))
            scaler.scale(full).backward()
            scaler.unscale_(opt)
            if corr is not None:
                gsum += torch.sqrt(sum(p.grad.float().pow(2).sum() for p in params))
                for p, c in zip(params, corr):
                    p.grad.add_(c)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()

            total += loss.item()
            steps += 1
    ratio = 0.0
    if corr is not None and steps > 0:
        cnorm = torch.sqrt(sum(c.float().pow(2).sum() for c in corr))
        ratio = (cnorm / (gsum / steps)).item()
    return total / max(steps, 1), steps, ratio


def control_variate(model, x_all, idx, sched, band, batches, batch_size):
    device = x_all.device
    params = list(model.parameters())
    lo, hi = band if band is not None else (0, sched.T)
    acc = [torch.zeros_like(p) for p in params]
    for _ in range(batches):
        pick = idx[torch.randint(0, len(idx), (batch_size,), device=device)]
        x0 = random_flip(x_all[pick])
        t = torch.randint(lo, hi, (batch_size,), device=device)
        eps = torch.randn_like(x0)
        grads = torch.autograd.grad(ddpm_loss(model, x0, t, eps, sched), params)
        for a, g in zip(acc, grads):
            a.add_(g, alpha=1.0 / batches)
    return acc


def fedavg(states, sizes):
    w = torch.tensor(sizes, dtype=torch.float64)
    w = w / w.sum()
    avg = {}
    for k, v in states[0].items():
        if v.is_floating_point():
            avg[k] = sum(wi.item() * s[k] for wi, s in zip(w, states))
        else:
            avg[k] = v.clone()
    return avg
