# Mutual Information Co-training

This repository is the source code for the paper:

**MICO: Selective Search with Mutual Information Co-training**

In Proceedings of the International Conference on Computational Linguistics (COLING) , 2022

*Zhanyu Wang, Xiao Zhang, Hyokun Yun, Choon Hui Teo and Trishul Chilimb*

## Introduction
This is the package of Mutual Information Co-training (MICO) for End2End topic sharding. MICO uses BERT to generate sentence representations, and performs query routing and document assignment with the representations. The document assignment module in MICO outputs almost equal-sized clusters, and the query routing module routes the queries to the cluster containing most (if not all) of its relevant documents. MICO achieves very high performance for topic sharding. 

This package can be tested through the example usage below.

## Usage
You can save the command below as a bash file and run it in the current folder. You can also find and run it in `./example/scripts/run_mico.sh`. It will take less than 5 minutes to finish running.

The results will be saved in `./results/`. In the folder `example_pair_BERT-finetune_layer-1_CLS_TOKEN_maxlen64_bs64_lr-bert5e-6_lr2e-4_warmup1000_entropy5_seed1` for this example experiment, we can see the final evaluation metrics saved in `metrics.json`. The document assigned to the clusters are saved in `clustered_docs.json` in a dictionary. The log files for training and evaluation are `*.log`. The model is saved as `*.pt`. The folder `./log` contains Tensorboard results for visualization. 

The `dataset_name` in the training command is set as `example` since we have an example dataset saved in `../example/data/example_dataset/`. You can change the `train_folder_path` and `test_folder_path` according to your needs.

During training, the `batch_size` is for each GPU card. If the current choice of `batch_size` is good on a machine with one GPU, we do not need to change it when switching to machines with more than one GPU (each with the same GPU memory). This is because we use the `DistributedDataParallel` function in `PyTorch` to support multi-GPU training: we assign one sub-process for each GPU and it maintains its own dataloader and counts its own epoch number (hence people usually focus on the iteration number instead of the epoch number). For a 4-GPU machine, finishing one epoch for each process means training the model for 4 epochs in total. For a GPU with 16GB memory, setting `batch_size=64` is good for the first try.

During testing, we use `DataParallel` in `PyTorch` for better efficiency (we only go through the dataset once with multi-GPU, much less than using `DistributedDataParallel`), and the `batch_size` is across all GPUs. Usually for testing, you can set a much larger `batch_size` than the one used in training, e.g., for four GPUs (each with 16GB memory), we can use `batch_size=2048`. You can also test the trained model directly by setting `--eval_only`.

    #!/bin/bash

    dataset_name=example
    train_folder_path=./example/data/${dataset_name}_train_csv/
    test_folder_path=./example/data/${dataset_name}_test_csv/

    batch_size=64
    selected_layer_idx=-1
    pooling_strategy=CLS_TOKEN
    max_length=64
    lr=2e-4
    lr_bert=5e-6
    entropy_weight=5
    num_warmup_steps=1000
    seed=1

    model_path=./example/results/${dataset_name}_pair_BERT-finetune_layer${selected_layer_idx}\
    _${pooling_strategy}\
    _maxlen${max_length}\
    _bs${batch_size}\
    _lr-bert${lr_bert}\
    _lr${lr}\
    _warmup${num_warmup_steps}\
    _entropy${entropy_weight}\
    _seed${seed}/

    python -u ./main.py \
        --model_path=${model_path} \
        --train_folder_path=${train_folder_path} \
        --test_folder_path=${test_folder_path} \
        --dim_input=768 \
        --number_clusters=64 \
        --dim_hidden=8 \
        --num_layers_posterior=0 \
        --batch_size=${batch_size} \
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



## Visualize results with Tensorboard
To visualize the curves of the metrics calculated during training and evaluation, please use Tensorboard (for `Pytorch` we use `TensorboardX` which is installed in the setting up section.) 

The results for each experiment is saved in the folder specified by `--model_path` in the bash commands. We also have log files in text format in that folder. After running the following command, you can open your browser and type `localhost:14095` to view the training results.

    # start tensorboard
    tensorboard --logdir=./results/ --port=14095 serve

## Memory profiling
Although we have adopted several techniques to decrease the memory usage, it is still possible that one encounters memory problem when running with large scale dataset. You can try this memory profiling method to estimate how much memory you will need for running MICO. 

Some tips: 
1. Setting `num_worker=0` is a good way to save memory and it almost does not affect the training speed. 
2. Running MICO on more GPUs will create more sub-process automatically, and each sub-process may consume much memory. Therefore, the memory usage increases linearly with the GPU number. If needed, you can set `export CUDA_VISIBLE_DEVICES=0` to only use 1 GPU in training to save memory.

To use the memory profiling method below, please make sure that the python package `memory_profiler` is installed. (If not, you can install it with `pip install memory_profiler`.) It can track the memory usage of the Python codes. For more details, please see https://pypi.org/project/memory-profiler/.

To use it to track the memory usage, you can try the command below.

    mprof run --interval=10 --multiprocess --include-children './your_bash_file.sh'

During the bash file running, you can plot the memory usage over time by the command below. Please replace `mprofile_***.dat` with the name of the profile results you want to plot (the lastest `dat` file will be used if the file is not specified). The figure will be saved as `memory_profile_result.png`.

    mprof plot -o memory_profile_result.png --backend agg mprofile_***.dat

## Setting up a new EC2 machine
For setting up a new EC2 machine to run the scripts, please use the codes below

    wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
    bash ./Anaconda3-2021.05-Linux-x86_64.sh
    source ~/.bashrc  
    conda install pytorch=1.7.1 cudatoolkit=9.2 -c pytorch
    pip install -r requirements.txt
    pip install memory_profiler

After download the data, you can replace the two folders (for training and testing data) in `./example/data/` by the two large scale datasets. Then, you can modify and run the script `./example/scripts/run_mico.sh`.

## License

This project is licensed under the Apache-2.0 License.
