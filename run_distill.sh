S=${1:-0}
source "$(dirname "$0")/common.sh"
A="$D --alpha $ALPHA --measure_alphas $ALPHA --local_epochs 5 --rounds 100 --seed $S --out_dir runs --wandb_project fed-diffusion --distill 1.0 --distill_n 1000 --distill_every 10 --distill_batch 32 --distill_start 10 --prox_ramp 20"

train() {
    n=$1${SUF}_s$S
    shift
    python -u measure_drift.py $A --name $n "$@" > logs/$n.log 2>&1
    for r in 25 50 100; do
        python fid.py $D --ckpt runs/${n}_round${r}_ema.pt --log fid.txt
    done
}

train distill_uniform --measure_rounds 25 50 100 &
train distill_adaptive --measure_rounds 10 20 25 30 40 50 60 70 80 90 100 --prox_adapt &
wait
