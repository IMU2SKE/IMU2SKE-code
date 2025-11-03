# 定义网格搜索参数
model_names=("LDT")
learning_rates=(3e-4)
epochs=(100)
seeds=(164)
use_types=("none" "use_skeleton" "use_imu" "use_skeleton,use_imu" "use_skeleton,use_imu,use_imu_skeleton" "use_skeleton,use_imu,use_multi")
attention_ratios=(0.0 1.0)  # 新增attention比例参数

# 执行网格搜索
for model_name in "${model_names[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for epoch in "${epochs[@]}"; do
            for seed in "${seeds[@]}"; do
                for use in "${use_types[@]}"; do
                    for ratio in "${attention_ratios[@]}"; do  # 新增attention比例循环
                        # 将组合中的逗号替换为空格
                        use_args=$(echo $use | tr ',' ' ')
                        echo "Training model: $model_name, lr: $lr, epochs: $epoch, seed: $seed, use: $use, attention_ratio: $ratio"
                        python Model/main.py \
                            --mode train \
                            --model $model_name \
                            --seed $seed \
                            --lr $lr \
                            --epochs $epoch \
                            --path ori/${model_name}_lr${lr}_epoch${epoch}_seed${seed}_${use//,/_}_ratio${ratio} \
                            --data-path PD_43.npy \
                            --cv True \
                            --mask None \
                            --use $use_args \
                            --attention_ratio $ratio
                    done
                done
            done
        done
    done
done