from urllib import parse
import numpy as np
from codes_q import create_bivariate_bicycle_codes
from stim_circ import dem_to_check_matrices, bb_mem_circuit
from codes_q import create_bivariate_bicycle_codes
import stim
import multiprocessing as mp
import argparse
import json


def decode_worker(d, dem, decoder_params, fail_target, sim_num, fail_num, lock, bposd=False):
    from mybp import FullBP
    from ldpc import BpOsdDecoder
    import os
    if bposd:
        bpd = BpOsdDecoder(
                decoder_params['pcm'],
                channel_probs=decoder_params['priors'],
                max_iter=1000,
                bp_method="ms",
                ms_scaling_factor=0,
                osd_method="osd_cs",
                osd_order=10
        )
    else:
        bpd = FullBP(**decoder_params)
    dem_sampler: stim.CompiledDemSampler = dem.compile_sampler(seed=os.getpid())
    CHUNK = 100

    while True:
        with lock:
            if fail_num.value >= fail_target:
                break
        
        det_data, obs_data, err_data = dem_sampler.sample(shots=CHUNK, return_errors=False, bit_packed=False)
        flgs = np.zeros(CHUNK, dtype=bool)
        decodeds = np.zeros((CHUNK, decoder_params["pcm"].shape[1]), dtype=int)
        for i in range(CHUNK):
            if bposd:
                flgs[i], decodeds[i] = True, bpd.decode(det_data[i])
            else:
                flgs[i], decodeds[i] = bpd.flip_decode(det_data[i])
        residual = (decodeds @ obs.T + obs_data) % 2
        with lock:
            fail_num.value += np.sum(np.any(residual, axis=1) | (flgs==0))
            sim_num.value += CHUNK
            if sim_num.value % 100000 ==0:
                pl = fail_num.value/sim_num.value/d
                error_bar = 1.96 * np.sqrt(pl*(1-pl)/sim_num.value)/d
                print(f"error: {fail_num.value}, shots: {sim_num.value:,}, p_L/round: {pl:.4e} ± {error_bar:.4e}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum error correction simulation")
    parser.add_argument("--d", type=int, default=12, help="Distance parameter (default: 12)")
    parser.add_argument("--num_processes", type=int, default=32, help="Number of processes (default: 32)")
    parser.add_argument("--l", type=int, default=12, help="l parameter (default: 12)")
    parser.add_argument("--m", type=int, default=6, help="m parameter (default: 6)")
    parser.add_argument("--p_list", type=str, default="[0.001, 0.002]", help="Physical error rates as JSON list (default: '[0.001, 0.002]')")
    parser.add_argument("--w_min", type=int, default=1, help="w_min parameter (default: 1)")
    parser.add_argument("--w_max", type=int, default=10, help="w_max parameter (default: 10)")
    parser.add_argument("--n_sample", type=int, default=10, help="n_sample parameter (default: 10)")
    parser.add_argument("--max_iter", type=int, default=100, help="max_iter parameter (default: 100)")
    parser.add_argument("--topk", type=int, default=50, help="topk parameter (default: 50)")
    parser.add_argument("--a_x", type=str, default="[3]", help="A values as JSON list (default: '[3]')")
    parser.add_argument("--a_ys", type=str, default="[1, 2]", help="A values as JSON list (default: '[1, 2]')")
    parser.add_argument("--b_y", type=str, default="[3]", help="A values as JSON list (default: '[3]')")
    parser.add_argument("--b_xs", type=str, default="[1, 2]", help="B values as JSON list (default: '[1, 2]')")
    parser.add_argument("--scheduling", type=str, default="parallel", help="Scheduling parameter (default: 'parallel')")
    parser.add_argument("--use_osd", type=bool, default=False, help="Use BP-OSD True/False" )
    
    args = parser.parse_args()
    
    d = args.d
    num_processes = args.num_processes
    l, m = args.l, args.m
    p_list = json.loads(args.p_list)
    w_min = args.w_min
    w_max = args.w_max
    n_sample = args.n_sample
    max_iter = args.max_iter
    topk = args.topk
    a_x = json.loads(args.a_x)
    a_ys = json.loads(args.a_ys)
    b_y = json.loads(args.b_y)
    b_xs = json.loads(args.b_xs)
    BBObj = create_bivariate_bicycle_codes(l, m, a_x, a_ys, b_xs, b_y) #Gross Code
    use_osd = args.use_osd  
    print("Using BP-OSD:", use_osd)

    n = 2 * l * m
    fail_target = 10
    code, A_list, B_list = BBObj
    if use_osd:
        fname = f"./data/bb_test_{l}_{m}_{d}_bposd.txt"
    else:
        fname = f"./data/bb_test_{l}_{m}_{d}_bpsf.txt"
    for p in p_list:
        print(f"Running p = {p}")
        circ = bb_mem_circuit(BBObj, n, d, None, p)
        dem = circ.detector_error_model()
        chk, obs, priors, col_dict = dem_to_check_matrices(dem, return_col_dict=True)
        manager = mp.Manager()
        fail_counter = manager.Value('i', 0)  # Counter for failures
        sim_counter = manager.Value('i', 0)  # Counter for total simulations
        lock = manager.Lock()
        processes = []
        decoder_params = {"pcm": chk, 
                            "i_max": max_iter, 
                            "priors": priors, 
                            "topk": topk, 
                            "w_min": w_min, 
                            "w_max": w_max, 
                            "n_sample": n_sample,
                            "scheduling": args.scheduling,
                            "max_procs": 1} # set to 1 to use serial decoding for each process
        for _ in range(num_processes):
            p_ = mp.Process(target=decode_worker, args=(d,dem, decoder_params, 
            fail_target, sim_counter, fail_counter, lock, use_osd))
            p_.start()
            processes.append(p_)

        # Wait for all processes to finish
        for p_ in processes:
            p_.join()

        # Report results
        pl = fail_counter.value/sim_counter.value
        print(f"error: {fail_counter.value}, shots: {sim_counter.value}," 
        f"p_L/round: {1 - (1 - pl) ** (1 / d):.4e}")
        with open(fname, "a") as f:
            f.write(f"physical error: {p}, error: {fail_counter.value}, shots: {sim_counter.value}," 
            f"p_L/round: {1 - (1 - pl) ** (1 / d):.4e}\n")

