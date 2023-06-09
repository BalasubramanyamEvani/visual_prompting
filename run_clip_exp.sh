#!/bin/bash

timestamp=$(date +%Y-%m-%d:%H:%M)

print_freq=10
save_freq=50
batch_size=256
num_workers=16
epochs=1000

optim="sgd"
learning_rate=40.0
weight_decay=0.0
warmup=1000
momentum=0.9
patience=1000

model="clip"
arch="vit_b32"
method="padding"
prompt_size=30

root="./data"
dataset="cifar10"
image_size=224

seed=0
model_dir="./out/${dataset}_${timestamp}/models"
image_dir="./out/${dataset}_${timestamp}/images"
filename=""
trial=1
resume=""
gpu=7

# Run script
python main_clip.py \
  --print_freq $print_freq \
  --save_freq $save_freq \
  --batch_size $batch_size \
  --num_workers $num_workers \
  --epochs $epochs \
  --optim "$optim" \
  --learning_rate $learning_rate \
  --weight_decay $weight_decay \
  --warmup $warmup \
  --momentum $momentum \
  --patience $patience \
  --model "$model" \
  --arch "$arch" \
  --method "$method" \
  --prompt_size $prompt_size \
  --root "$root" \
  --dataset "$dataset" \
  --image_size $image_size \
  --seed $seed \
  --model_dir "$model_dir" \
  --image_dir "$image_dir" \
  --filename "$filename" \
  --trial $trial \
  --resume "$resume" \
  --gpu $gpu \
  --use_wandb \
  # --evaluate
