# W&B credentials come from the environment (WANDB_API_KEY) or `wandb login`
export WANDB_API_KEY=${WANDB_API_KEY:?set WANDB_API_KEY before launching}

accelerate launch src/f5_tts/train/finetune_cli.py --logger wandb --dataset_name ro_tts --num_warmup_updates 50 --last_per_updates 2000 --log_samples --learning_rate 1e-4 --save_per_updates 300 --batch_size_per_gpu 16384 --max_samples 128 --keep_last_n_checkpoints 3 --epochs 50 --checkpoint_path ro_tts_lang_mod
