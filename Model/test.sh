python Model/main.py \
    --mode test \
    --model ContrastSTGCN \
    --seed 164 \
    --lr 1e-4 \
    --epochs 1 \
    --path cstgcn/001 \
    --data-path data/toy_dataset/train_subset_toy.npy \
    --cv True \
    --mask None

python Model/lib/vote.py cstgcn 001 test