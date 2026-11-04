import argparse
import time

import torch
from torchmetrics.image.fid import FrechetInceptionDistance

from data import load_cifar10
from diffusion import Schedule
from sample import build, generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--model", default="small", choices=["small", "standard"])
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--batch", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--swap_ckpt", default=None)
    parser.add_argument("--swap_range", type=int, nargs=2, default=None)
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--log", default="fid_results.txt")
    parser.add_argument("--experts", default=None)
    parser.add_argument("--expert_bands", type=int, nargs="+", default=[0, 1000])
    parser.add_argument("--n_clients", type=int, default=10)
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--wandb_id", default=None)
    parser.add_argument("--wandb_project", default="timestep-drift")
    parser.add_argument("--round", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sched = Schedule(device=device)
    model, name = build(args, device, sched)
    t0 = time.time()

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    real, _ = load_cifar10(args.data_root)
    for i in range(0, len(real), args.batch):
        fid.update(((real[i:i + args.batch] + 1) / 2).to(device), real=True)
    print(f"real features done | {time.time() - t0:.0f}s", flush=True)

    done = 0
    while done < args.n:
        b = min(args.batch, args.n - done)
        fid.update(generate(model, b, sched, device, args.seed + done, args.batch), real=False)
        done += b
        print(f"generated {done}/{args.n} | {time.time() - t0:.0f}s", flush=True)

    score = fid.compute().item()
    print(f"{name} | n {args.n} | FID {score:.3f} | {time.time() - t0:.0f}s")
    with open(args.log, "a") as f:
        extra = f"\troute {model.counts.tolist()}" if hasattr(model, "counts") else ""
        f.write(f"{name}\t{args.n}\t{score:.4f}{extra}\n")
    if args.wandb_id:
        import wandb
        wandb.init(project=args.wandb_project, id=args.wandb_id, resume="allow")
        wandb.define_metric("fid_round")
        wandb.define_metric("fid/*", step_metric="fid_round")
        key = "fid/ema" if name.endswith("_ema") else "fid/raw"
        wandb.log({key: score, "fid_round": args.round})
        wandb.finish()


if __name__ == "__main__":
    main()
