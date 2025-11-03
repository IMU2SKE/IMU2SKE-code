# 定义网格搜索参数
model_names=("IMU2SKE_v6_mask_5fold")
learning_rates=(5e-3)  # 增加更多学习率选项
epochs=(100)  # 增加更多epoch选项，覆盖短、中、长训练周期
seeds=(0)  # 增加更多随机种子选项
freeze_encoders=(False)
clips=(True)
dual_clips=(True)
imu_clips=(True)
skeleton_clips=(True)
pretrain_paths=(random)
skeleton_pretrain_paths=(random)

# 执行网格搜索
for model_name in "${model_names[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for epoch in "${epochs[@]}"; do
            for seed in "${seeds[@]}"; do
                for freeze in "${freeze_encoders[@]}"; do
                    for pretrain in "${pretrain_paths[@]}"; do
                        for skeleton_pretrain in "${skeleton_pretrain_paths[@]}"; do
                            for clip in "${clips[@]}"; do
                                for dual_clip in "${dual_clips[@]}"; do
                                    for imu_clip in "${imu_clips[@]}"; do
                                        for skeleton_clip in "${skeleton_clips[@]}"; do
                                            echo "Training model: $model_name, lr: $lr, epochs: $epoch, seed: $seed, freeze_encoder: $freeze, pretrain_path: $pretrain, skeleton_pretrain_path: $skeleton_pretrain"
                                            output_dir=results/train_results/exp/${model_name}_lr${lr}_epoch${epoch}_seed${seed}_freeze${freeze}_pretrain${pretrain//\//_}_skeleton${skeleton_pretrain//\//_}_clip${clip}_dualclip${dual_clip}_imuclip${imu_clip}_skeletonclip${skeleton_clip}
                                            mkdir -p "$output_dir"
                                            python Model/imu_finetune.py \
                                                --mode train \
                                                --model $model_name \
                                                --seed $seed \
                                                --lr $lr \
                                                --epochs $epoch \
                                                --path exp/${model_name}_lr${lr}_epoch${epoch}_seed${seed}_freeze${freeze}_pretrain${pretrain//\//_}_skeleton${skeleton_pretrain//\//_}_clip${clip}_dualclip${dual_clip}_imuclip${imu_clip}_skeletonclip${skeleton_clip} \
                                                --data-path PD_43.npy \
                                                --cv True \
                                                --mask None \
                                                --freeze-encoder $freeze \
                                                --pretrain-path $pretrain \
                                                --skele-path $skeleton_pretrain \
                                                --clip $clip \
                                                --dual-clip $dual_clip \
                                                --imu-clip $imu_clip \
                                                --skeleton-clip $skeleton_clip  > results/train_results/exp/${model_name}_lr${lr}_epoch${epoch}_seed${seed}_freeze${freeze}_pretrain${pretrain//\//_}_skeleton${skeleton_pretrain//\//_}_clip${clip}_dualclip${dual_clip}_imuclip${imu_clip}_skeletonclip${skeleton_clip}/nomask_log.txt
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
