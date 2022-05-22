#!/bin/bash
#SBATCH -p speech-gpu

python generate_seq2seq.py \
    --debug 1 \
    --gen_prefix "dev_best_bs${2}" \
    --ckpt_name best.ckpt \
    --model_dir $1 \
    --n_epoch 1 \
    --batch_size 1 \
    --train_path fd_dev.json\
    --dev_path fd_dev.json\
    --test_path fd_test.json\
    --vocab_file bpe.10k.vocab \
    --beam_size $2 \
    --max_tot_src_len 14336 \
    --max_src_len 128 \
    --max_tgt_len 1024 \
    --max_gen_len 1024 \
    --min_gen_len 20
