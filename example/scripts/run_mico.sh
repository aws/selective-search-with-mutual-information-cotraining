#!/bin/bash

dataset_name=example
train_folder_path=../data/${dataset_name}_train_csv/
test_folder_path=../data/${dataset_name}_test_csv/

batch_size=64
batch_size_test=512
selected_layer_idx=-1
pooling_strategy=CLS_TOKEN
max_length=64
lr=2e-4
lr_bert=5e-6
entropy_weight=5
num_warmup_steps=1000
seed=1

model_path=../results/${dataset_name}_pair_BERT-finetune_layer${selected_layer_idx}\
_${pooling_strategy}\
_maxlen${max_length}\
_bs${batch_size}\
_lr-bert${lr_bert}\
_lr${lr}\
_warmup${num_warmup_steps}\
_entropy${entropy_weight}\
_seed${seed}/

python -u ../../main.py \
    --model_path=${model_path} \
    --train_folder_path=${train_folder_path} \
    --test_folder_path=${test_folder_path} \
    --is_csv_header \
    --dim_input=768 \
    --number_clusters=64 \
    --dim_hidden=8 \
    --num_layers_posterior=0 \
    --batch_size=${batch_size} \
    --batch_size_test=${batch_size_test} \
    --lr=${lr} \
    --num_warmup_steps=${num_warmup_steps} \
    --lr_prior=0.1 \
    --num_steps_prior=1 \
    --init=0.0 \
    --clip=1.0 \
    --epochs=1 \
    --log_interval=10 \
    --check_val_test_interval=10000 \
    --save_per_num_epoch=100 \
    --num_bad_epochs=10 \
    --seed=${seed} \
    --entropy_weight=${entropy_weight} \
    --num_workers=0 \
    --cuda \
    --lr_bert=${lr_bert} \
    --max_length=${max_length} \
    --pooling_strategy=${pooling_strategy} \
    --selected_layer_idx=${selected_layer_idx} 
