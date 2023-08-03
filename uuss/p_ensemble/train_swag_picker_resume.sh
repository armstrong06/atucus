#!/bin/bash

ensemble_N=3  
pretrain_models=(scsn_seed2412045_model003.pt scsn_seed2412046_model004.pt scsn_seed2412047_model003.pt)
swag_seeds=(1 2 3)

# ${BS}_${SGD_LR}_${WD}_${SWA_LR}_${EPOCHS}_${SWA_START}_${K}
# 128_0.0005_5e-3_0.0002_84_24_20
batch_size=128
sgd_lr=0.0005
weight_decay=5e-3
swa_lr=0.0002
epochs=86
swa_start=61
K=20

save_freq=1

N=$(($ensemble_N-1))
#echo $N
for e in $(seq 0 $N); do
        seed=${swag_seeds[$e]}
        echo "Using seed $seed"
        echo "Using pretrained model $pt_model"
        dir="./seed${seed}_${batch_size}_${sgd_lr}_${weight_decay}_${swa_lr}_61_26_${K}"
        echo $dir
        python -u ../../swag_modified/train/run_swag.py --data_path="../uuss_data/p_resampled" \
                --train_dataset="uuss_train_4s_1dup.h5" --validation_dataset="uuss_validation_4s_1dup.h5" \
                --swa_resume="used_models/seed${seed}_swag-61.pt" --eval_freq=1 --max_num_models=$K --seed=$seed \
                --batch_size=$batch_size --epochs=$epochs --model="PPicker" --save_freq=$save_freq --lr_init=$sgd_lr \
                --wd=$weight_decay --swa --swa_start=$swa_start --swa_lr=$swa_lr --cov_mat --dir=$dir --no_schedule &
        wait -n 
        wait
done
