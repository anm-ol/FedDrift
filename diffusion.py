import torch


class Schedule:
    """Noise schedule indexed by an integer t in [0, T); larger t = noisier.

    kind="ddpm": linear-beta DDPM, x_t = sqrt(ab) x0 + sqrt(1-ab) eps, model predicts eps.
    kind="fm":   rectified flow, x_t = (1-tau) x0 + tau eps with tau = (t+1)/T, model predicts
                 the velocity eps - x0. Same integer t is fed to the UNet, so time bands,
                 drift buckets and RegMean bands are shared code (they are equal-width in tau).
    """

    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device="cpu", kind="ddpm"):
        self.T, self.kind = T, kind
        if kind == "ddpm":
            self.betas = torch.linspace(beta_start, beta_end, T, device=device)
            self.alphas = 1.0 - self.betas
            self.alphabar = torch.cumprod(self.alphas, dim=0)
            self.sqrt_ab = self.alphabar.sqrt()
            self.sqrt_1mab = (1.0 - self.alphabar).sqrt()
        elif kind == "fm":
            self.tau = torch.arange(1, T + 1, device=device, dtype=torch.float32) / T
            self.sqrt_ab = 1.0 - self.tau  # signal coefficient
            self.sqrt_1mab = self.tau      # noise coefficient
            self.alphabar = self.sqrt_ab.pow(2)
        else:
            raise ValueError(kind)
        self.snr = self.sqrt_ab.pow(2) / self.sqrt_1mab.pow(2)


def q_sample(x0, t, eps, sched):
    return sched.sqrt_ab[t].view(-1, 1, 1, 1) * x0 + sched.sqrt_1mab[t].view(-1, 1, 1, 1) * eps


def target(x0, eps, sched):
    """Regression target of the network: eps (DDPM) or velocity eps - x0 (flow matching)."""
    return eps if sched.kind == "ddpm" else eps - x0


def ddpm_loss(model, x0, t, eps, sched):
    # objective-agnostic despite the name (kept for the existing call sites)
    xt = q_sample(x0, t, eps, sched)
    return (model(xt, t) - target(x0, eps, sched)).pow(2).mean()


@torch.no_grad()
def ddpm_sample(model, shape, sched, device, gen=None, clip=None):
    """Ancestral DDPM sampling. clip=True uses the posterior q(x_{t-1} | x_t, x0) with x0
    clipped to [-1, 1] (Ho et al.'s sampler). Without it, 64x64 samples drift in per-channel
    mean from t~700 on and come out as saturated single-colour images. Default: clip only
    above 32px, so CIFAR-10 numbers stay comparable with the earlier unclipped runs."""
    if sched.kind == "fm":
        return fm_sample(model, shape, sched, device, gen)
    if clip is None:
        clip = shape[-1] > 32
    x = torch.randn(shape, device=device, generator=gen)
    for i in reversed(range(sched.T)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        eps = model(x, t).float()
        if clip:
            ab, ab_prev = sched.alphabar[i], (sched.alphabar[i - 1] if i > 0 else torch.ones((), device=device))
            x0 = ((x - sched.sqrt_1mab[i] * eps) / sched.sqrt_ab[i]).clamp(-1, 1)
            mean = (ab_prev.sqrt() * sched.betas[i] * x0 + sched.alphas[i].sqrt() * (1 - ab_prev) * x) / (1 - ab)
            var = sched.betas[i] * (1 - ab_prev) / (1 - ab)
        else:
            mean = (x - sched.betas[i] / sched.sqrt_1mab[i] * eps) / sched.alphas[i].sqrt()
            var = sched.betas[i]
        noise = torch.randn(shape, device=device, generator=gen) if i > 0 else torch.zeros_like(x)
        x = mean + var.sqrt() * noise
    return x


@torch.no_grad()
def fm_sample(model, shape, sched, device, gen=None):
    # Euler on dx/dtau = v from tau=1 to 0, one step per t index (same 1000 NFE as DDPM)
    x = torch.randn(shape, device=device, generator=gen)
    for i in reversed(range(sched.T)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        v = model(x, t).float()
        tau_next = sched.tau[i - 1] if i > 0 else 0.0
        x = x + (tau_next - sched.tau[i]) * v
    return x
