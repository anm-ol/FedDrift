# Sourced by run_*.sh. Override with env vars, e.g.
#   DATASET=celeba64 OBJECTIVE=fm bash run_regmean.sh 0
# CelebA has no classes (8 attribute groups for 10 clients), so it defaults to Dirichlet 0.1.
DATASET=${DATASET:-cifar10}
OBJECTIVE=${OBJECTIVE:-ddpm}
if [ "$DATASET" = celeba64 ]; then ALPHA=${ALPHA:-0.1}; else ALPHA=${ALPHA:-class}; fi
# run-name suffix; empty for the original cifar10/ddpm/class setup so old names still match
SUF=""
[ "$DATASET" != cifar10 ] && SUF="${SUF}_$DATASET"
[ "$OBJECTIVE" != ddpm ] && SUF="${SUF}_$OBJECTIVE"
[ "$ALPHA" != class ] && SUF="${SUF}_a$ALPHA"
D="--dataset $DATASET --objective $OBJECTIVE"
mkdir -p logs runs
python make_partitions.py --dataset $DATASET --alphas $ALPHA --seed $S > /dev/null
