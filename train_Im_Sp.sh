#!/bin/bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
python train_Im_Sp.py \
--checkpoint_dir ./data/checkpoints/IM_Speech \
--temp_dir ./tmp_eval/IM_Speech \
--project IM_Speech \
--architecture git-large-coco \
--batch_size 16 \
--eval_step 5000 \
--lr 5e-5 \
--gpu 0,1,2,3 \
--update_frequency 1 \
--start_epoch 0 \
--vit_fix \
--warmup \
--warmup_iteration 10000 \
--tot_iters 100000 \
--distributed \