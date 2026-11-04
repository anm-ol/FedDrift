import argparse
import os

import torch
import torch.nn as nn
from torchvision.utils import save_image

from diffusion import Schedule, ddpm_sample, q_sample
from model import UNet


class Swap(nn.Module):
    def __init__(self, base, other, lo, hi):
        super().__init__()
        self.base, self.other, self.lo, self.hi = base, other, lo, hi

    def forward(self, x, t):
        return self.other(x, t) if self.lo <= t[0].item() < self.hi else self.base(x, t)


class Routed(nn.Module):
    def __init__(self, glob, clients, bands, sched, draws=8, seed=0, look=400):
        super().__init__()
        self.glob, self.clients, self.bands, self.sched, self.draws = glob, nn.ModuleList(clients), bands, sched, draws
        self.seed, self.look = seed, look
        self.route = None
        self.counts = torch.zeros(len(clients), dtype=torch.long)

    def expert(self, t):
        return any(lo <= t < hi for lo, hi in self.bands)

    @torch.no_grad()
    def pick(self, x, t):
        gen = torch.Generator(device=x.device).manual_seed(self.seed + int(self.counts.sum()))
        if t == self.sched.T - 1:
            return torch.randint(0, len(self.clients), (x.shape[0],), device=x.device, generator=gen)
        s = self.sched
        tc = min(t, self.look)
        for i in range(t, tc, -1):
            eps = self.glob(x, torch.full((x.shape[0],), i, device=x.device, dtype=torch.long)).float()
            x = (x - s.betas[i] / s.sqrt_1mab[i] * eps) / s.alphas[i].sqrt()
            x = x + s.betas[i].sqrt() * torch.randn(x.shape, device=x.device, generator=gen)
        t = tc
        x0 = (x - s.sqrt_1mab[t] * self.glob(x, torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)).float())
        x0 = (x0 / s.sqrt_ab[t]).clamp(-1, 1)
        err = torch.zeros(len(self.clients), x.shape[0], device=x.device)
        for _ in range(self.draws):
            tt = torch.randint(0, t, (x.shape[0],), device=x.device, generator=gen)
            eps = torch.randn(x.shape, device=x.device, generator=gen)
            xt = q_sample(x0, tt, eps, self.sched)
            for i, c in enumerate(self.clients):
                err[i] += (c(xt, tt).float() - eps).pow(2).mean(dim=(1, 2, 3))
        return err.argmin(dim=0)

    def forward(self, x, t):
        ti = t[0].item()
        if ti == self.sched.T - 1:
            self.route = None
        if not self.expert(ti):
            return self.glob(x, t)
        if self.route is None:
            self.route = self.pick(x, ti)
            self.counts += torch.bincount(self.route.cpu(), minlength=len(self.clients))
        out = torch.zeros(x.shape, device=x.device)
        for i, c in enumerate(self.clients):
            m = self.route == i
            if m.any():
                out[m] = c(x[m], t[m]).float()
        return out


def load(path, size, device):
    model = UNet(size).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.eval()


def build(args, device, sched=None):
    model = load(args.ckpt, args.model, device)
    name = os.path.splitext(args.ckpt)[0]
    if getattr(args, "experts", None):
        clients = [load(f"{args.experts}_client{i}.pt", args.model, device) for i in range(args.n_clients)]
        b = args.expert_bands
        bands = [(b[i], b[i + 1]) for i in range(0, len(b), 2)]
        model = Routed(model, clients, bands, sched, args.draws, args.seed)
        name = args.experts + "_exp" + "_".join(f"{lo}-{hi}" for lo, hi in bands)
    if args.swap_ckpt:
        lo, hi = args.swap_range
        model = Swap(model, load(args.swap_ckpt, args.model, device), lo, hi)
        name += f"_swap{lo}-{hi}"
    return model, name


def generate(model, n, sched, device, seed=0, batch=1000):
    gen = torch.Generator(device=device).manual_seed(seed)
    out = []
    for i in range(0, n, batch):
        b = min(batch, n - i)
        with torch.autocast(device.type, enabled=(device.type == "cuda")):
            x = ddpm_sample(model, (b, 3, 32, 32), sched, device, gen)
        out.append((x.clamp(-1, 1) + 1) / 2)
    return torch.cat(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--model", default="small", choices=["small", "standard"])
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--swap_ckpt", default=None)
    parser.add_argument("--swap_range", type=int, nargs=2, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sched = Schedule(device=device)
    model, name = build(args, device)

    x = generate(model, args.n, sched, device, args.seed)
    out = args.out or name + "_samples.png"
    save_image(x, out, nrow=int(args.n ** 0.5), padding=2)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
