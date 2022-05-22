#!/bin/bash
#SBATCH -p speech-gpu

python train_seq2seq.py \
    --debug 0\
    --save_prefix neural_baseline_fd \
    --model_type longformer \
    --n_epoch 100 \
    --train_path fd_train.json \
    --dev_path fd_dev.json\
    --test_path fd_dev.json\
    --vocab_file bpe.10k.vocab \
    --batch_size 1 \
    --eval_batch_size 2 \
    --warmup_steps 150 \
    --use_gelu 1 \
    --num_encoder_layer 3 \
    --num_decoder_layer 6 \
    --learning_rate 1e-4 \
    --feedforward_hidden_size 4 \
    --hidden_size 512 \
    --max_tot_src_len 14336 \
    --max_src_len 128 \
    --max_tgt_len 1024 \
    --gradient_accumulation_steps 200 \
    --print_every 1 \
    --save_every 9999999999999999 \
    --eval_every 30
