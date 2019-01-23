#!/bin/bash

# input arguments
DATA="DD"  # MUTAG, ENZYMES, NCI1, NCI109, DD, PTC, PROTEINS, COLLAB, IMDBBINARY, IMDBMULTI
# general settings
gm=DGCNN_DS  # model: DGCNN, DGCNN_RNN, DGCNN_DS, DGCNN_LSTM,
gpu_or_cpu=gpu
GPU=0  # select the GPU number
CONV_SIZE="32-32-32"
sortpooling_k=150  # If k <= 1, then k is set to an integer so that k% of graphs have nodes less than this integer
FP_LEN=128  # final dense layer's input dimension, decided by data
n_hidden=97  #97 final dense layer's hidden size
bsize=50  # batch size
dropout=True
learning_rate=0.0001
num_patience=50
num_epochs=500

CUDA_VISIBLE_DEVICES=${GPU} python main.py \
    -seed 1 \
    -data $DATA \
    -num_epochs $num_epochs \
    -num_patience $num_patience \
    -hidden $n_hidden \
    -latent_dim $CONV_SIZE \
    -sortpooling_k $sortpooling_k \
    -out_dim $FP_LEN \
    -batch_size $bsize \
    -gm $gm \
    -mode $gpu_or_cpu \
    -dropout $dropout \
    -learning_rate $learning_rate

