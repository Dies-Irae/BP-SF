#!/bin/bash
export PYTHONPATH=./minimal_bp_decoder/src_python:$PYTHONPATH
# Example script to run test.py with command-line arguments
echo "Speed Bechmanrk for [[144,12,12]] memory experiment (CUDAQ BP1000-OSD10)"
python benchmark.py \
    --d 12 \
    --num_processes 1 \
    --l 12 \
    --m 6 \
    --p_list "[0.001, 0.002, 0.003]" \
    --w_min 1 \
    --w_max 10 \
    --n_sample 10 \
    --max_iter 100 \
    --topk 50 \
    --a_x "[3]" \
    --a_ys "[1, 2]" \
    --b_xs "[1, 2]" \
    --b_y "[3]" \
    --decoder_type "cudaq" \

echo "Speed Bechmanrk for [[144,12,12]] memory experiment (BPSF, GPU Estimation)"
python gpu_est.py \
    --d 12 \
    --num_processes 1 \
    --l 12 \
    --m 6 \
    --p_list "[0.001, 0.002, 0.003]" \
    --w_min 1 \
    --w_max 10 \
    --n_sample 10 \
    --max_iter 100 \
    --topk 50 \
    --a_x "[3]" \
    --a_ys "[1, 2]" \
    --b_xs "[1, 2]" \
    --b_y "[3]" \

echo "Speed Bechmanrk for [[144,12,12]] memory experiment (BP)"
python benchmark.py \
    --d 12 \
    --num_processes 1 \
    --l 12 \
    --m 6 \
    --p_list "[0.001, 0.002, 0.003]" \
    --w_min 1 \
    --w_max 10 \
    --n_sample 10 \
    --max_iter 100 \
    --topk 50 \
    --a_x "[3]" \
    --a_ys "[1, 2]" \
    --b_xs "[1, 2]" \
    --b_y "[3]" \
    --decoder_type "bp" \

echo "Speed Bechmanrk for [[144,12,12]] memory experiment (BPSF, P=8)"
python benchmark.py \
    --d 12 \
    --num_processes 8 \
    --l 12 \
    --m 6 \
    --p_list "[0.001, 0.002, 0.003]" \
    --w_min 1 \
    --w_max 10 \
    --n_sample 10 \
    --max_iter 100 \
    --topk 50 \
    --a_x "[3]" \
    --a_ys "[1, 2]" \
    --b_xs "[1, 2]" \
    --b_y "[3]" \
    --decoder_type "bpsf" \

echo "Speed Bechmanrk for [[144,12,12]] memory experiment (BPSF, P=1)"
python benchmark.py \
    --d 12 \
    --num_processes 1 \
    --l 12 \
    --m 6 \
    --p_list "[0.001, 0.002, 0.003]" \
    --w_min 1 \
    --w_max 10 \
    --n_sample 10 \
    --max_iter 100 \
    --topk 50 \
    --a_x "[3]" \
    --a_ys "[1, 2]" \
    --b_xs "[1, 2]" \
    --b_y "[3]" \
    --decoder_type "bpsf" \


echo "Speed Bechmanrk for [[144,12,12]] memory experiment (BP1000-OSD10)"
python benchmark.py \
    --d 12 \
    --num_processes 1 \
    --l 12 \
    --m 6 \
    --p_list "[0.001, 0.002, 0.003]" \
    --w_min 1 \
    --w_max 10 \
    --n_sample 10 \
    --max_iter 100 \
    --topk 50 \
    --a_x "[3]" \
    --a_ys "[1, 2]" \
    --b_xs "[1, 2]" \
    --b_y "[3]" \
    --decoder_type "bposd" \



