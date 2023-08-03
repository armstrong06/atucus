#!/bin/bash

batch_sizes=(128)
sgd_lrs=(0.003)
#swa_lrs=(0.00005 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1)
swa_lrs=(0.001)
weight_decays=(5e-3)
epochs=(85)
swa_start=(25)

#10 - 5, 20 - 10, 15, 30 - 20, 25
# --resume="SGD/128_0.005_3e-4_11/checkpoint-10.pt"
for BS in ${batch_sizes[@]}; do
    for SGD_LR in ${sgd_lrs[@]}; do
        for WD in ${weight_decays[@]}; do
            for SWA_LR in ${swa_lrs[@]}; do
                for EPOCHS in ${epochs[@]}; do
                    for SWA_START in ${swa_start[@]}; do
                        dir="SWAG_tune_lr/${BS}_${SGD_LR}_${WD}_${SWA_LR}_${EPOCHS}_${SWA_START}_20"
                        echo $dir
                        python -u ../../swag_modified/train/run_swag.py --data_path="../uuss_data/s_resampled" \
                                --train_dataset="uuss_train_6s_1dup.h5" --validation_dataset="uuss_validation_6s_1dup.h5" \
                                --resume="SGD_tuning/128_0.003_5e-3_30/checkpoint-25.pt" --eval_freq=1 --max_num_models=20 \
                                --batch_size=$BS --epochs=$EPOCHS --model="SPicker" --save_freq=1 --lr_init=$SGD_LR \
                                --wd=$WD --swa --swa_start=$SWA_START --swa_lr=$SWA_LR --cov_mat --dir=$dir --no_schedule &
                        wait -n
                        wait
                    done
                done    
            done
        done
    done
done
