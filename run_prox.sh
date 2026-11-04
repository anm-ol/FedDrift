S=${1:-0}
A="--alpha class --measure_alphas class --local_epochs 5 --rounds 100 --seed $S --out_dir runs --wandb_project fed-diffusion --prox 0.4 --prox_ramp 20"
mkdir -p logs runs
[ -f partitions/cifar10_K10_class_seed$S.json ] || python make_partitions.py --alphas class --seed $S

train() {
    n=$1_s$S
    shift
    python -u measure_drift.py $A --name $n "$@" > logs/$n.log 2>&1
    for r in 25 50 100; do
        python fid.py --ckpt runs/${n}_round${r}_ema.pt --log fid.txt
    done
}

train prox_uniform --measure_rounds 25 50 100 &
train prox_adaptive --measure_rounds 10 20 25 30 40 50 60 70 80 90 100 --prox_adapt &
wait
