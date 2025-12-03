#!/bin/bash
export PYTHONPATH=./minimal_bp_decoder/src_python:$PYTHONPATH
# Example script to run test.py with command-line arguments
echo ""
echo "Running [[126,12,10]] memory experiment"
python cbb_test.py \
    --d 10 \
    --num_processes 32 \
    --l 7 \
    --m 9 \
    --p_list "[0.001, 0.002, 0.003, 0.005, 0.007, 0.01]" \
    --batch_size 10000 \
    --w_min 1 \
    --w_max 10 \
    --n_sample 10 \
    --max_iter 100 \
    --topk 50 \
    --p1 "[(0, 0), (1, 1), (58, 58)]" \
    --p2 "[(0, 0), (13, 13), (41, 41)]" 
echo ""

echo ""
echo "Running [[154,6,16]] memory experiment"
python cbb_test.py \
    --d 16 \
    --num_processes 32 \
    --l 7 \
    --m 11 \
    --p_list "[0.001, 0.002, 0.003, 0.005, 0.007, 0.01]" \
    --batch_size 10000 \
    --w_min 1 \
    --w_max 10 \
    --n_sample 10 \
    --max_iter 100 \
    --topk 50 \
    --p1 "[(0, 0), (1, 1), (31, 31)]" \
    --p2 "[(0, 0), (19, 19), (53, 53)]" 
echo ""

echo "Running [[288,12,18]] memory experiment"
python test.py \
    --d 18 \
    --num_processes 32 \
    --l 12 \
    --m 12 \
    --p_list "[0.002, 0.003, 0.004]" \
    --batch_size 10000 \
    --w_min 1 \
    --w_max 10 \
    --n_sample 10 \
    --max_iter 100 \
    --topk 50 \
    --a_x "[3]" \
    --a_ys "[2, 7]" \
    --b_xs "[1, 2]" \
    --b_y "[3]" \
    --scheduling "serial"

echo ""
echo "Running [[72,12,6]] memory experiment"
python test.py \
    --d 6 \
    --num_processes 32 \
    --l 6 \
    --m 6 \
    --p_list "[1e-4, 2e-4, 1e-3, 2e-3, 5e-3, 1e-2]" \
    --batch_size 10000 \
    --w_min 1 \
    --w_max 4 \
    --n_sample 5 \
    --max_iter 100 \
    --topk 20 \
    --a_x "[3]" \
    --a_ys "[1, 2]" \
    --b_xs "[1, 2]" \
    --b_y "[3]"

echo ""
echo "Running [[144,12,12]] memory experiment"
python test.py \
    --d 12 \
    --num_processes 32 \
    --l 12 \
    --m 6 \
    --p_list "[0.001, 0.002, 0.003, 0.005, 0.01]" \
    --batch_size 10000 \
    --w_min 1 \
    --w_max 10 \
    --n_sample 10 \
    --max_iter 100 \
    --topk 50 \
    --a_x "[3]" \
    --a_ys "[1, 2]" \
    --b_xs "[1, 2]" \
    --b_y "[3]"


