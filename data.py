import json
import os

import numpy as np
import torch
from torchvision import datasets

CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

DATASETS = ["cifar10", "celeba64", "tinyimagenet"]
IMG_SIZE = {"cifar10": 32, "celeba64": 64, "tinyimagenet": 64}
# CelebA has no classes; clients are skewed over the 2^3 groups of these attributes
CELEBA_ATTRS = ("Male", "Smiling", "Black_Hair")
# CIFAR-10 has 50k training images; the other datasets are subsampled to match, so
# client sizes and local steps per round stay comparable across datasets
N_TRAIN = 50000
N_TEST = 10000


def load_cifar10(root="./data", train=True):
    ds = datasets.CIFAR10(root, train=train, download=True)
    x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    labels = np.array(ds.targets)
    return x, labels


def load_cifar100(root="./data", train=True):
    ds = datasets.CIFAR100(root, train=train, download=True)
    x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    return x, np.array(ds.targets)


def load_dataset(name, root="./data", train=True):
    """Images in [-1, 1] as float [N, 3, H, W] and integer labels (classes or CelebA attribute groups)."""
    if name == "cifar10":
        return load_cifar10(root, train)
    path = os.path.join(root, f"{name}_{'train' if train else 'test'}.npz")
    if not os.path.exists(path):
        os.makedirs(root, exist_ok=True)
        x, labels = (_fetch_celeba if name == "celeba64" else _fetch_tinyimagenet)(train)
        np.savez(path + ".tmp.npz", x=x, labels=labels)
        os.replace(path + ".tmp.npz", path)
    d = np.load(path)
    return torch.from_numpy(d["x"]).permute(0, 3, 1, 2).float() / 127.5 - 1.0, d["labels"]


def _fetch_celeba(train):
    # flwrlabs/celeba is ~12GB; stream and keep only what we use
    from datasets import load_dataset as hf_load
    from PIL import Image
    n = N_TRAIN if train else N_TEST
    split = "train" if train else "test"
    # rows are sorted by identity, so a prefix would hold only a few thousand celebrities:
    # stream the whole split and keep a seeded uniform subset of rows
    total = {"train": 162770, "test": 19962}[split]
    keep = np.zeros(total, dtype=bool)
    keep[np.random.default_rng(0).choice(total, min(n, total), replace=False)] = True
    ds = hf_load("flwrlabs/celeba", split=split, streaming=True)
    xs, labels = [], []
    for i, ex in enumerate(ds):
        if i >= total:
            break
        if not keep[i]:
            continue
        im = ex["image"].convert("RGB")
        w, h = im.size  # 178x218 aligned faces; standard 140px center crop, then 64x64
        l, t = (w - 140) // 2, (h - 140) // 2
        xs.append(np.asarray(im.crop((l, t, l + 140, t + 140)).resize((64, 64), Image.LANCZOS)))
        labels.append(sum(int(ex[a]) << i for i, a in enumerate(reversed(CELEBA_ATTRS))))
        if len(xs) == n:
            break
    return np.stack(xs), np.array(labels)


def _fetch_tinyimagenet(train):
    from datasets import load_dataset as hf_load
    ds = hf_load("zh-plus/tiny-imagenet", split="train" if train else "valid")
    labels = np.array(ds["label"])
    keep = np.arange(len(labels))
    if train and len(keep) > N_TRAIN:  # stratified subsample: 250 images per class
        rng = np.random.default_rng(0)
        per = N_TRAIN // len(np.unique(labels))
        keep = np.sort(np.concatenate([rng.permutation(np.flatnonzero(labels == c))[:per] for c in np.unique(labels)]))
    xs = np.stack([np.asarray(ds[int(i)]["image"].convert("RGB")) for i in keep])
    return xs, labels[keep]


def random_flip(x):
    flip = torch.rand(x.shape[0], device=x.device) < 0.5
    return torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)


def dirichlet_partition(labels, n_clients, alpha, seed, min_size=256, max_tries=10000):
    rng = np.random.default_rng(seed)

    if alpha is None:
        perm = rng.permutation(len(labels))
        return [np.sort(p) for p in np.array_split(perm, n_clients)]

    if alpha == "class":
        classes = np.unique(labels)
        if len(classes) % n_clients:
            raise ValueError(f"class split needs #classes ({len(classes)}) divisible by #clients ({n_clients}); use a Dirichlet alpha")
        # one class per client for CIFAR-10; C/K classes per client otherwise
        return [np.flatnonzero(np.isin(labels, classes[k::n_clients])) for k in range(n_clients)]

    for _ in range(max_tries):
        parts = [[] for _ in range(n_clients)]
        for c in np.unique(labels):
            idx = rng.permutation(np.flatnonzero(labels == c))
            p = rng.dirichlet(alpha * np.ones(n_clients))
            cuts = (np.cumsum(p)[:-1] * len(idx)).astype(int)
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k].append(chunk)
        parts = [np.sort(np.concatenate(p)) for p in parts]
        if min(len(p) for p in parts) >= min_size:
            return parts
    raise RuntimeError(f"could not get every client above {min_size} samples")


def count_matrix(labels, parts):
    return np.stack([np.bincount(labels[p], minlength=labels.max() + 1) for p in parts])


def print_counts(counts, title=""):
    print(title)
    if counts.shape[1] != len(CLASSES):
        for k, row in enumerate(counts):
            print(f"client {k:>2}: {row.sum():>6} images, {(row > 0).sum():>3} classes, top-class share {row.max() / row.sum():.2f}")
        print()
        return
    print("client " + "".join(f"{c[:5]:>6}" for c in CLASSES) + "   total")
    for k, row in enumerate(counts):
        print(f"{k:>6} " + "".join(f"{v:>6}" for v in row) + f"{row.sum():>8}")
    print()


def alpha_tag(alpha):
    if alpha is None:
        return "iid"
    return alpha if alpha == "class" else f"{alpha:g}"


def parse_alpha(s):
    if s == "iid":
        return None
    return s if s == "class" else float(s)


def partition_path(out_dir, n_clients, alpha, seed, dataset="cifar10"):
    tag = alpha_tag(alpha)
    tag = tag if tag in ("iid", "class") else f"alpha{tag}"
    return f"{out_dir}/{dataset}_K{n_clients}_{tag}_seed{seed}.json"


def save_partition(path, parts, alpha, seed):
    with open(path, "w") as f:
        json.dump({"alpha": alpha, "seed": seed, "parts": [p.tolist() for p in parts]}, f)


def load_partition(path):
    with open(path) as f:
        return [np.array(p) for p in json.load(f)["parts"]]
