S=${1:-0}
A="--alpha class --measure_alphas class --local_epochs 5 --rounds 150 --seed $S --out_dir runs --wandb_project fed-diffusion --measure_rounds 25 50 100 150 --share_size 1000"
mkdir -p logs runs
[ -f partitions/cifar10_K10_class_seed$S.json ] || python make_partitions.py --alphas class --seed $S

train() {
    n=$1_s$S
    shift
    python -u measure_drift.py $A --name $n "$@" > logs/$n.log 2>&1
    for r in 100 150; do
        python fid.py --ckpt runs/${n}_round${r}_ema.pt --log fid.txt
    done
}

train shared_all_t --share_batch 32 --share_range 0 1000 &
train shared_low_t --share_batch 32 --share_range 0 400 &
train shared_low_t_13 --share_batch 13 --share_range 0 400 &
train shared_mid_t --share_batch 32 --share_range 400 800 &
wait
