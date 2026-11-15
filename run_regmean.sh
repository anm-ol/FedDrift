S=${1:-0}
source "$(dirname "$0")/common.sh"
A="$D --alpha $ALPHA --measure_alphas $ALPHA --local_epochs 5 --rounds 100 --seed $S --out_dir runs --wandb_project fed-diffusion --measure_rounds 25 50 75 100 --regmean unif --regmean_a 0.9 --regmean_n 256 --sample_every 10"

python -u measure_drift.py $A --name regmean_unif${SUF}_s$S --regmean_rho unif > logs/regmean_unif${SUF}_s$S.log 2>&1 &
python -u measure_drift.py $A --name regmean_adaptive${SUF}_s$S --regmean_rho gen > logs/regmean_adaptive${SUF}_s$S.log 2>&1 &
wait
for n in regmean_unif${SUF}_s$S regmean_adaptive${SUF}_s$S; do
    for r in 25 50 75 100; do
        python fid.py $D --ckpt runs/${n}_round${r}_ema.pt --log fid.txt
    done
done
