S=${1:-0}
A="--alpha class --measure_alphas class --local_epochs 5 --rounds 100 --seed $S --out_dir runs --wandb_project fed-diffusion --measure_rounds 25 50 75 100 --regmean unif --regmean_a 0.9 --regmean_n 256 --sample_every 10"
mkdir -p logs runs
[ -f partitions/cifar10_K10_class_seed$S.json ] || python make_partitions.py --alphas class --seed $S

python -u measure_drift.py $A --name regmean_unif_s$S --regmean_rho unif > logs/regmean_unif_s$S.log 2>&1 &
python -u measure_drift.py $A --name regmean_adaptive_s$S --regmean_rho gen > logs/regmean_adaptive_s$S.log 2>&1 &
wait
for n in regmean_unif_s$S regmean_adaptive_s$S; do
    for r in 25 50 75 100; do
        python fid.py --ckpt runs/${n}_round${r}_ema.pt --log fid.txt
    done
done
