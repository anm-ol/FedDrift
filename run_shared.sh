S=${1:-0}
source "$(dirname "$0")/common.sh"
A="$D --alpha $ALPHA --measure_alphas $ALPHA --local_epochs 5 --rounds 150 --seed $S --out_dir runs --wandb_project fed-diffusion --measure_rounds 25 50 100 150 --share_size 1000"

train() {
    n=$1${SUF}_s$S
    shift
    python -u measure_drift.py $A --name $n "$@" > logs/$n.log 2>&1
    for r in 100 150; do
        python fid.py $D --ckpt runs/${n}_round${r}_ema.pt --log fid.txt
    done
}

train shared_all_t --share_batch 32 --share_range 0 1000 &
train shared_low_t --share_batch 32 --share_range 0 400 &
train shared_low_t_13 --share_batch 13 --share_range 0 400 &
train shared_mid_t --share_batch 32 --share_range 400 800 &
wait
