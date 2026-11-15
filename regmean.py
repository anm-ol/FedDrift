import torch
import torch.nn as nn

from diffusion import q_sample


def layers(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


@torch.no_grad()
def grams(model, x, sched, bands, n, batch, gen):
    model.eval()
    mods = layers(model)
    out = {name: [] for name, _ in mods}
    cur, cnt = {}, {}

    def hook(name):
        def f(m, inp):
            a = inp[0].float()
            if a.dim() == 4:
                a = a.movedim(1, -1)
            a = a.reshape(-1, a.shape[-1])
            cur[name] = cur.get(name, 0) + a.T @ a
            cnt[name] = cnt.get(name, 0) + a.shape[0]
        return f

    handles = [m.register_forward_pre_hook(hook(name)) for name, m in mods]
    for lo, hi in bands:
        cur.clear()
        cnt.clear()
        for i in range(0, n, batch):
            b = min(batch, n - i)
            idx = torch.randint(0, len(x), (b,), device=x.device, generator=gen)
            t = torch.randint(lo, hi, (b,), device=x.device, generator=gen)
            eps = torch.randn((b, *x.shape[1:]), device=x.device, generator=gen)
            model(q_sample(x[idx], t, eps, sched), t)
        for name in out:
            out[name].append(cur[name] / cnt[name])
    for h in handles:
        h.remove()
    return {name: torch.stack(v) for name, v in out.items()}


def regmean(ds, gs, pi, a, rel):
    gs = [a * g + (1 - a) * torch.diag(torch.diagonal(g)) for g in gs]
    S = sum(p * g for p, g in zip(pi, gs))
    S = S + rel * S.diagonal().mean() * torch.eye(len(S), device=S.device)
    N = sum(p * (d.movedim(1, -1) @ g) for p, d, g in zip(pi, ds, gs))
    X = torch.linalg.solve(S, N.reshape(-1, S.shape[0]).T).T
    return X.reshape(N.shape).movedim(-1, 1)
