#!/bin/bash

# python -u swag_modified/uncertainty/uncertainty.py --file="tuning/128_0.05_3e-4_0.01/swag-20.pt" \
#         --data_path="./data" --train_dataset="uuss_train.h5" --validation_dataset="uuss_test_fewerhist.h5" \
#         --n_duplicates_train=3 --batch_size=128 --method=SWAG --cov_mat --scale=0.5 \
#         --save_path="tuning/128_0.05_3e-4_0.01/swag_test_uncertainty"

batch_size=128
sgd_lr=0.003
weight_decay=5e-3
swa_lr=0.001
epochs=56
swa_start=25
K=20
seeds=(1 2 3)
splits=("test_fewerhist" "validation" "NGB")

for split in ${splits[@]}; do
for seed in ${seeds[@]}; do
        dir="./seed${seed}_${batch_size}_${sgd_lr}_${weight_decay}_${swa_lr}_${epochs}_${swa_start}_${K}"
        #dir="./seed${seed}_${batch_size}_${sgd_lr}_${weight_decay}_${swa_lr}"
        echo $dir
        python -u ../../swag_modified/uncertainty/uncertainty.py --file="${dir}/swag-${epochs}.pt"\
                --data_path="../uuss_data/s_resampled" --train_dataset="uuss_train_6s_1dup.h5" \
		--validation_dataset="uuss_${split}_6s_1dup.h5" --model="SPicker" \
                --batch_size=${batch_size} --method="SWAG" --cov_mat --scale=0.5 --seed=${seed} \
                --save_path="${dir}/eval/swag_${split}_uncertainty_50" --max_num_models=$K --N=50 &
        wait -n 
        wait
done
done
