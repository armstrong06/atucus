#!/bin/bash

#["BatchSize", "SGD_lr", "WD", "Mom" "SWA_lr", "K"]
batch_sizes=(128)
sgd_lrs=(0.001 0.003 0.005 0.007 0.01 0.03 0.05)
weight_decays=(1e-4 5e-4 1e-3 5e-3 1e-2)
epochs=(30)
#10 - 5, 20 - 10, 15, 30 - 20, 25

for BS in ${batch_sizes[@]}; do
    for SGD_LR in ${sgd_lrs[@]}; do
        for WD in ${weight_decays[@]}; do
            for EPOCHS in ${epochs[@]}; do
                dir="./SGD_tuning/${BS}_${SGD_LR}_${WD}_${EPOCHS}"
                echo $dir
                python -u ../../swag_modified/train/run_swag.py --data_path="../uuss_data/s_resampled" \
                        --train_dataset="uuss_train_6s_1dup.h5" --validation_dataset="uuss_validation_6s_1dup.h5" \
                        --load_model="../../pretrained_models/stead_seed2412045_model_004.pt" --eval_freq=1 \
                        --batch_size=$BS --epochs=$EPOCHS --model="SPicker" --save_freq=1 --lr_init=$SGD_LR \
                        --wd=$WD --dir=$dir --no_schedule &
                wait -n
                wait
            done
        done
    done
done
