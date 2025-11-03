# 定义模型名称和参数
model_names=("LDT")
learning_rates=(3e-4)
epochs=(100 200)
seeds=(164)
use_types=("none" "use_skeleton" "use_imu" "use_skeleton,use_imu" "use_skeleton,use_imu,use_imu_skeleton" "use_skeleton,use_imu,use_multi")

# 执行验证
for model_name in "${model_names[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for epoch in "${epochs[@]}"; do
            for seed in "${seeds[@]}"; do
                for use in "${use_types[@]}"; do
                    # 将组合中的逗号替换为空格
                    use_args=$(echo $use | tr ',' ' ')
                    echo "Validating model: $model_name, lr: $lr, epochs: $epoch, seed: $seed, use: $use"
                    python Model/main.py \
                        --mode val \
                        --model $model_name \
                        --seed $seed \
                        --lr $lr \
                        --epochs $epoch \
                        --path ori/${model_name}_lr${lr}_epoch${epoch}_seed${seed}_${use//,/_} \
                        --data-path PD_43.npy \
                        --cv True \
                        --mask None \
                        --use $use_args
                done
            done
        done
    done
done