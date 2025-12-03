from myldpc.bp_decoder import BpDecoder
from mybp import FullBP
import numpy as np
from bposd.css import css_code
import multiprocessing as mp
import os 

def BB_code(ell, m, p1, p2):
    # returns the parity check matrices of a bivariate bicycle code arXiv:2308.07915
    # p1 = (a,b,c), p2 = (d,e,f)
    # A = x^a + y^b + y^c
    # B = y^d + x^e + x^f
    a, b, c = p1
    d, e, f = p2
    # define cyclic shift matrices
    I_ell = np.identity(ell, dtype=int)
    I_m = np.identity(m, dtype=int)
    x = np.kron(np.roll(I_ell, 1, axis=1), I_m)
    y = np.kron(I_ell, np.roll(I_m, 1, axis=1))

    # define parity check matrices
    A = (np.linalg.matrix_power(x, a) + np.linalg.matrix_power(y, b) + np.linalg.matrix_power(y, c)) % 2
    B = (np.linalg.matrix_power(y, d) + np.linalg.matrix_power(x, e) + np.linalg.matrix_power(x, f)) % 2

    AT = np.transpose(A)
    BT = np.transpose(B)

    hx = np.hstack((A, B))
    hz = np.hstack((BT, AT))

    return hx, hz

def CBB_code(ell, m, p1, p2):
    # l,m are two primes
    # p1 = (a,b,c), p2 = (d,e,f)
    # A = \sum xy^p1[i]
    # B = \sum xy^p2[i]
    # define cyclic shift matrices
    I_ell = np.identity(ell, dtype=int)
    I_m = np.identity(m, dtype=int)
    x = np.kron(np.roll(I_ell, 1, axis=1), I_m)
    y = np.kron(I_ell, np.roll(I_m, 1, axis=1))

    A = np.zeros([ell*m, ell*m])
    B = np.zeros_like(A)
    # define parity check matrices
    for i in p1:
        A += np.linalg.matrix_power(x@y, i)
    for i in p2:
        B += np.linalg.matrix_power(x@y, i)
    A = A % 2
    B = B % 2
    AT = np.transpose(A)
    BT = np.transpose(B)

    hx = np.hstack((A, B))
    hz = np.hstack((BT, AT))

    return hx, hz

def generate_sample(N, p):
    error_x = np.zeros(N, dtype=int)
    error_z = np.zeros(N, dtype=int)
    for i in range(N):
        rand = np.random.random()
        if rand < p/3:
            error_z[i] = 1
            error_x[i] = 0
        elif p/3<= rand< 2*p/3:
            error_z[i] = 0
            error_x[i] = 1
        elif 2*p/3 <= rand < p:
            error_z[i] = 1
            error_x[i] = 1
        else:
            error_z[i] = 0
            error_x[i] = 0

    return error_x, error_z


def test_thread(chunk_size, p, qcode, bpd_z, bpd_x, dec_type):
    fail = 0
    for i in range(chunk_size):
        error_x, error_z = generate_sample(qcode.N, p)
        synd_z = qcode.hx @ error_z % 2
        synd_x = qcode.hz @ error_x % 2
        if dec_type == 'my':
            flgz, out_z = bpd_z.flip_decode(synd_z)
            flgx, out_x = bpd_x.flip_decode(synd_x)
            residual_x = (error_x + out_x) % 2
            residual_z = (error_z + out_z) % 2
            flag = flgz and flgx
        else:
            correctionz = bpd_z.decode(synd_z)
            correctionx = bpd_x.decode(synd_x)
            residual_x = (error_x + correctionx) % 2
            residual_z = (error_z + correctionz) % 2
            flag = bpd_z.converge and bpd_x.converge
        
        if not flag or (qcode.lz @ residual_x % 2).any() or (qcode.lx @ residual_z
                % 2).any():
            fail += 1
    return fail


def worker(fail_counter, sim_counter, fail_target, lock, chunk_size, p, qcode, bpd_z, bpd_x, dec_type):
    """
    Worker function to run test_thread until the failure target is reached.

    Args:
        fail_counter (mp.Value): Shared counter for failures.
        sim_counter (mp.Value): Shared counter for total simulations.
        fail_target (int): Target number of failures.
        lock (mp.Lock): Lock for synchronizing shared variables.
        chunk_size (int): Number of samples per thread execution.
        p (float): Error probability.
        qcode (css_code): Quantum code object.
        bpd_z (MS): Belief propagation decoder for Z errors.
        bpd_x (MS): Belief propagation decoder for X errors.
    """
    np.random.seed(os.getpid())
    while True:
        with lock:
            if fail_counter.value >= fail_target:
                break
        # Run test_thread and get the number of failures
        fails = test_thread(chunk_size, p, qcode, bpd_z, bpd_x, dec_type)
        with lock:
            fail_counter.value += fails
            sim_counter.value += chunk_size
            if sim_counter.value % 100000 == 0:
                ler = fail_counter.value/sim_counter.value
                se = np.sqrt(ler*(1-ler)/sim_counter.value)
                ci = 1.96*se
                print(f"Process {os.getpid()}: {sim_counter.value} simulations, {fail_counter.value} failures, LER: {ler:.4e}, error bar: ±{ci:.4e}")


def main(num_processes, fail_target, chunk_size, p, hx, hz, qcode, dec_type, m_iter, topk, w_min, w_max, n_sample):
    """
    Main function to run the simulation for a given error rate.

    Args:
        num_processes (int): Number of parallel processes.
        fail_target (int): Target number of failures.
        chunk_size (int): Number of samples per thread execution.
        p (float): Error probability.
    """
    if dec_type == 'my':
        bpd_x = FullBP(hz, m_iter, p=p, topk=topk, w_min=w_min, w_max=w_max, n_sample=n_sample)
        bpd_z = FullBP(hx, m_iter, p=p, topk=topk, w_min=w_min, w_max=w_max, n_sample=n_sample)
    else:
        bpd_x = BpDecoder(
                qcode.hz,  # the z-type parity check matrix
                error_rate=p,
                channel_probs=[None],
                max_iter=m_iter,  # max number of iterations for BP
                bp_method="ms",
                ms_scaling_factor=0,
            )
        bpd_z = BpDecoder(
                qcode.hx,  # the z-type parity check matrix
                error_rate=p,
                channel_probs=[None],
                max_iter=m_iter,  # max number of iterations for BP
                bp_method="ms",
                ms_scaling_factor=0,
            )
    
    # Shared variables for multiprocessing
    manager = mp.Manager()
    fail_counter = manager.Value('i', 0)  # Counter for failures
    sim_counter = manager.Value('i', 0)  # Counter for total simulations
    lock = manager.Lock()

    # Start worker processes
    processes = []
    for _ in range(num_processes):
        p_ = mp.Process(target=worker, args=(fail_counter, sim_counter, fail_target, lock,
                                             chunk_size, p, qcode, bpd_z, bpd_x, dec_type))
        p_.start()
        processes.append(p_)

    # Wait for all processes to finish
    for p_ in processes:
        p_.join()

    # Report results
    print(f"Error rate: {p}, Total failures: {fail_counter.value}, Total simulations: {sim_counter.value}")
    return fail_counter.value, sim_counter.value

if __name__ == "__main__":
    # Set parameters
    num_processes = 20
    fail_target = 100
    chunk_size = 100
    decoder = 'my' # "BP" or "my"
    max_iter = 50
    topk = 8
    w_min = 1
    w_max = 1
    n_sample = 8
    # Generate the quantum code and decoders
    # hx, hz = BB_code(12, 12, [3,2,7], [3,1,2])
    hx, hz = CBB_code(7, 11, [0, 1, 31], [0, 19, 53])
    qcode = css_code(hx, hz)
    qcode.test()
    if decoder != 'my':
        fname = str(qcode.N) + decoder + f"{max_iter}.txt"
    else:
        fname = str(qcode.N) + decoder + f"{max_iter}.txt"
    f = open(fname, "a")
    error_rates = [0.015, 0.03, 0.05, 0.07, 0.09, 0.11]
    simulated = []
    errors = []
    ler = []
    for p in error_rates:
        print("Starting simulation for error rate:", p)
        fail, sim = main(num_processes, fail_target, chunk_size, p, hx, hz, qcode, decoder, max_iter, topk, w_min, w_max, n_sample)
        simulated.append(sim)
        errors.append(fail)
        ler.append(fail/sim)
    f.write("Error rate: " + str(error_rates) + "\n")
    f.write("Simulated: " + str(simulated) + "\n")
    f.write("Errors: " + str(errors) + "\n")
    f.write("LER: " + str(ler) + "\n")
    f.close()