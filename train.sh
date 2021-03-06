accelerate launch src/train.py \
  --model_type concat \
  --backbone electra \
  --embedding_method cls_with_pooler \
  --train_data_path ./data/train_full.tsv \
  --max_length 512 \
  --pad_to_max_length \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-5 \
  --weight_decay 0.0 \
  --llrd_factor 0.8 \
  --num_train_epochs 3 \
  --gradient_accumulation_steps 2 \
  --num_warmup_ratio 0.1 \
  --classifier_dropout_rate 0.1 \
  --reinit_layers 0 \
  --noise_lambda 0.0 \
  --lr_scale_factor 1.0 \
  --seed 42 \
  --output_dir ./model \
  --log_dir ./log