#!/bin/bash

ensemble_N=3  
pretrain_models=(stead_seed2412045_model_004.pt  stead_seed2412046_model_004.pt  stead_seed2412047_model_003.pt)
swag_seeds=(1 2 3)

# ${BS}_${SGD_LR}_${WD}_${SWA_LR}_${EPOCHS}_${SWA_START}_${K}
# 128_0.003_5e-3_0.001_85_25_20

batch_size=128
sgd_lr=0.003
weight_decay=5e-3
swa_lr=0.001
epochs=56
swa_start=25
K=20

save_freq=1

N=$(($ensemble_N-1))
#echo $N
for e in $(seq 0 $N); do
        seed=${swag_seeds[$e]}
        pt_model=${pretrain_models[$e]}
        echo "Using seed $seed"
        echo "Using pretrained model $pt_model"
        dir="./seed${seed}_${batch_size}_${sgd_lr}_${weight_decay}_${swa_lr}_${epochs}_${swa_start}_${K}"
        echo $dir
        python -u ../../swag_modified/train/run_swag.py --data_path="../uuss_data/s_resampled" \
            --train_dataset="uuss_train_6s_1dup.h5" --validation_dataset="uuss_validation_6s_1dup.h5" \
	        --load_model="../../pretrained_models/${pt_model}" --eval_freq=1 --max_num_models=$K --seed=$seed \
            --batch_size=$batch_size --epochs=$epochs --model="SPicker" --save_freq=$save_freq --lr_init=$sgd_lr \
            --wd=$weight_decay --swa --swa_start=$swa_start --swa_lr=$swa_lr --cov_mat --dir=$dir --no_schedule &
        wait -n 
        wait
done
