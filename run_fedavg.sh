S=${1:-0}
A="--alpha class --measure_alphas class --local_epochs 5 --rounds 100 --seed $S --out_dir runs --wandb_project fed-diffusion"
mkdir -p logs runs
[ -f partitions/cifar10_K10_class_seed$S.json ] || python make_partitions.py --alphas class --seed $S

n=fedavg_s$S
python -u measure_drift.py $A --name $n --measure_rounds 10 25 50 100 > logs/$n.log 2>&1
for r in 25 50 100; do
    python fid.py --ckpt runs/${n}_round${r}_ema.pt --log fid.txt
done
