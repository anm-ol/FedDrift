S=${1:-0}
source "$(dirname "$0")/common.sh"
A="$D --alpha $ALPHA --measure_alphas $ALPHA --local_epochs 5 --rounds 100 --seed $S --out_dir runs --wandb_project fed-diffusion"

n=fedavg${SUF}_s$S
python -u measure_drift.py $A --name $n --measure_rounds 10 25 50 100 > logs/$n.log 2>&1
for r in 25 50 100; do
    python fid.py $D --ckpt runs/${n}_round${r}_ema.pt --log fid.txt
done
