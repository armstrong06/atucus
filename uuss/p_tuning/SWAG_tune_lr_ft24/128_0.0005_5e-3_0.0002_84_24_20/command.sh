../../swag_modified/train/run_swag.py --data_path=../uuss_data/p_resampled --train_dataset=uuss_train_4s_1dup.h5 --validation_dataset=uuss_validation_4s_1dup.h5 --resume=SGD_tuning/128_0.0005_5e-3_30/checkpoint-24.pt --eval_freq=1 --max_num_models=20 --batch_size=128 --epochs=84 --model=PPicker --save_freq=1 --lr_init=0.0005 --wd=5e-3 --swa --swa_start=24 --swa_lr=0.0002 --cov_mat --dir=SWAG_tune_lr_ft24/128_0.0005_5e-3_0.0002_84_24_20 --no_schedule