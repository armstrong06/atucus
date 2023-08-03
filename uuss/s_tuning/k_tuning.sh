#!/bin/bash

#["BatchSize", "SGD_lr", "WD", "Mom" "SWA_lr", "K"]
# 128_0.005_3e-4_0.001_60_0_20

BS=128
SGD_LR=0.005
SWA_LR=0.001
WD=3e-4
EPOCHS=45
SWA_START=10

max_num_models=(20)
#10 - 5, 20 - 10, 15, 30 - 20, 25
# --resume="SGD/128_0.005_3e-4_11/checkpoint-10.pt"
#--load_model="../pretrained_models/scsn_seed2412045_model003.pt"
for K in ${max_num_models[@]}; do
    dir="SWAG_tune/${BS}_${SGD_LR}_${WD}_${SWA_LR}_${EPOCHS}_${SWA_START}_${K}"
    echo $dir
    python -u ../../swag_modified/train/run_swag.py --data_path="../uuss_data/p_resampled" \
            --train_dataset="uuss_train_4s_1dup.h5" --validation_dataset="uuss_validation_4s_1dup.h5" \
	    --resume="SGD/128_0.005_3e-4_11/checkpoint-10.pt" --eval_freq=1 --max_num_models=$K\
            --batch_size=$BS --epochs=$EPOCHS --model="PPicker" --save_freq=1 --lr_init=$SGD_LR \
            --wd=$WD --swa --swa_start=$SWA_START --swa_lr=$SWA_LR --cov_mat --dir=$dir --no_schedule &
    wait -n
    wait
done
