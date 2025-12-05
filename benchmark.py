import numpy as np
from codes_q import create_bivariate_bicycle_codes
from build_circuit import build_circuit, dem_to_check_matrices
from codes_q import create_bivariate_bicycle_codes
import stim
import time
from mybp import FullBP
from ldpc import BpOsdDecoder
from ldpc import BpDecoder
import cudaq_qec as qec
import argparse
import json

def getTrainingData(circ, numShots):
    dem = circ.detector_error_model()
    chk, obs, priors, col_dict = dem_to_check_matrices(
        dem, return_col_dict=True)

    dem_sampler: stim.CompiledDemSampler = dem.compile_sampler(seed=1234)
    det_data, obs_data, err_data = dem_sampler.sample(
        shots=numShots, return_errors=False, bit_packed=False)

    return numShots, det_data, obs_data,chk,obs,priors

def test_sample(decoder_type, decoder, samples, obs_data, obs, chk):
    flgs = np.zeros(samples.shape[0])
    decoded = np.zeros((samples.shape[0], chk.shape[1]))
    if decoder_type == "bpsf":
        times = []
        print("TESTING BP-SF")
        for i in range(samples.shape[0]):
            time_start = time.time()
            decoder.flip_decode(samples[i])
            time_end = time.time()
            times.append(time_end - time_start)
        print("Ave time per sample: ", 1000 * sum(times)/len(times), "ms")

    elif decoder_type == "bp":
        print("TESTING BP")
        times = []
        for i in range(samples.shape[0]):
            time_start = time.time()
            decoder.decode(samples[i])
            time_end = time.time()
            times.append(time_end - time_start)
        print("Ave time per sample: ", 1000 * sum(times)/len(times), "ms")
    
    elif decoder_type == "cudaq":
        print("TESTING CUDAQ BPOSD")
        times = []
        for i in range(samples.shape[0]):
            time_start = time.time()
            nvResults = decoder.decode(samples[i])
            time_end = time.time()
            times.append(time_end - time_start)
        print("Ave time per sample: ", 1000 * sum(times)/len(times), "ms")

    elif decoder_type == "bposd":
        print("TESTING BPOSD")
        times = []
        for i in range(samples.shape[0]):
            time_start = time.time()
            decoder.decode(samples[i])
            time_end = time.time()
            times.append(time_end - time_start)
        print("Ave time per sample: ", 1000 * sum(times)/len(times), "ms")
    
    return 1000 * sum(times)/len(times)

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
    parser.add_argument("--decoder_type", type=str, default="bp", help="Decoder type (default: 'bp')" )

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
    decoder_type = args.decoder_type  
    code, A_list, B_list = BBObj
    shots = 20000
    for p in p_list:
        print(f"Physical error rate: {p}")
        circ = build_circuit(code, A_list, B_list, 
                                            p=p, # physical error rate
                                            num_repeat=d, # usually set to code distance
                                            z_basis=True,   # whether in the z-basis or x-basis
                                            use_both=False, # whether use measurement results in both basis to decode one basis
                                        )
        numShots, det_data, obs_data,chk,obs,priors = getTrainingData(circ, shots)
        if decoder_type == "bpsf": 
            my_bpd = FullBP(chk, max_iter, priors, topk=topk, w_min=w_min, w_max=w_max, n_sample=n_sample, max_procs=num_processes, scheduling=args.scheduling)
            print("start testing bpsf")
            avg_time = test_sample("bpsf", my_bpd, det_data, obs_data, obs, chk)
            fname = f"./data/time_{decoder_type}_P={num_processes}.txt"
            with open(fname, "a") as f:
                f.write(f"{p}, {avg_time} ms\n")
        elif decoder_type == "bposd":
            bposd = BpOsdDecoder(
                            chk,
                            channel_probs=priors,
                            max_iter=1000,
                            bp_method="ms",
                            ms_scaling_factor=0,
                            osd_method="osd_cs",
                            osd_order=10
                        )
            print("start testing bposd")
            avg_time = test_sample("bposd", bposd, det_data, obs_data, obs, chk)
            fname = f"./data/time_{decoder_type}.txt"
            with open(fname, "a") as f:
                f.write(f"{p}, {avg_time} ms\n")
            
        elif decoder_type == "bp":
            bp = BpDecoder(
                            chk,
                            channel_probs=priors,
                            max_iter=max_iter,
                            bp_method="ms",
                            ms_scaling_factor=0,
                        )
            print("start testing bp")
            avg_time = test_sample("bp", bp, det_data, obs_data, obs, chk)
            fname = f"./data/time_{decoder_type}.txt"
            with open(fname, "a") as f:
                f.write(f"{p}, {avg_time} ms\n")
        elif decoder_type == "cudaq":
            opts = dict() 
            opts['error_rate_vec'] = priors
            opts['max_iterations']=1000
            opts['use_osd']=True
            opts['osd_method']=3
            opts['osd_order']=10
            opts['use_sparsity']=True
            chkDenseForNV = np.array(chk.todense(order='C'))
            nvdec = qec.get_decoder('nv-qldpc-decoder', chkDenseForNV, **opts)
            print("start testing CUDAQ BPOSD Single")
            avg_time = test_sample("cudaq", nvdec, det_data, obs_data, obs, chk)
            fname = f"./data/time_{decoder_type}.txt"
            with open(fname, "a") as f:
                f.write(f"{p}, {avg_time} ms\n")


