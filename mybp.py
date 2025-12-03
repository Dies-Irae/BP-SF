import itertools
import numpy as np
from myldpc.bp_decoder import BpDecoder
import random
import math
import scipy.sparse as sp
import multiprocessing as mp
from queue import Empty
import cudaq_qec as qec


def compute_syndrome_sparse(pcm_csc, test_positions):
    """Compute syndrome for sparse test vector efficiently"""
    syndrome_cols = pcm_csc[:, test_positions]
    syndrome = np.array(syndrome_cols.sum(axis=1) % 2, dtype=np.uint8)
    return syndrome.flatten()

    


def decode_worker(pcm, pcm_csc, priors, i_max, input_queue, output_queue, solution_found_event):
    # Initialize decoder for this worker process
    bpd = BpDecoder(
        pcm,  # the parity check matrix
        error_channel=priors,  # the error rate on each bit
        max_iter=i_max,  # the maximum iteration depth for BP
        bp_method="ms",  # BP method.
        ms_scaling_factor=0
    )
    while True:
        job = input_queue.get()
        if job is None: # sentinel to stop worker
            break
        
        syndrome, test_pos_list, serial_num = job

        for test_pos in test_pos_list:
            if solution_found_event.is_set():
                break # Another worker found a solution, stop processing this chunk.

            test_pos = np.array(test_pos)
            test_syndrome = compute_syndrome_sparse(pcm_csc, test_pos)
            new_syndrome = np.remainder(syndrome + test_syndrome, 2)
            decoded = bpd.decode(new_syndrome)
            flg = bpd.converge
            if flg:
                decoded[test_pos] = 1 - decoded[test_pos]
                output_queue.put((True, decoded, serial_num))
                break # Solution found, no need to check other combinations in this chunk.
        
        # Signal that this chunk is done, whether a solution was found or not.
        output_queue.put((False, None, serial_num))


def sample_n_choose_k(iterable, k, num_samples):
        """
        Samples k elements from an iterable n times.

        Args:
            iterable: The input iterable (e.g., list, tuple).
            k: The number of elements to choose in each combination.
            num_samples: The number of samples to generate.

        Returns:
            A list of tuples, where each tuple is a combination of k elements.
        """
        if k > len(iterable):
            raise ValueError("k cannot be greater than the length of the iterable")
        
        # Convert numpy array to list if necessary
        if isinstance(iterable, np.ndarray):
            iterable = iterable.tolist()
        
        num_comb = math.comb(len(iterable), k)
        if num_samples >= num_comb:
            all_combinations = list(itertools.combinations(iterable, k))
            return all_combinations
        
        if num_samples/num_comb < 0.001:
            # random sampling will very unlikely to generate
            # the same combination twice, so we can use it for large iterable
            samples = []
            for _ in range(num_samples):
                selected = random.sample(iterable, k)
                samples.append(tuple(selected))
            return samples
        else:
            all_combinations = list(itertools.combinations(iterable, k))
            return random.sample(all_combinations, num_samples)
    

class FullBP:
    def __init__(self, pcm, i_max, priors=None, p=None, w_min=0, w_max=0, n_sample=0, topk=0, max_procs=1, scheduling="parallel"):
        """
        Initialize the Min-Sum decoder.

        Args:
            pcm (np.ndarray): Parity-check matrix.
            i_max (int): Maximum number of iterations.
            priors (np.ndarray): Prior probabilities for each bit. 
            p (float): Physical error rate (capacity model). 
            w_min (int): Minimum weight of the flipping vector.
            w_max (int): Maximum weight of the flipping vector.
            n_sample (int): Number of samples to generate for each combination.
            topk (int): Number of top bits to consider for flipping.
            max_procs (int): Maximum number of processes to use for parallel decoding.
        """
        self.H_csc = pcm.tocsc()
        if isinstance(pcm, (sp.csr_matrix, sp.csc_matrix)):
            self.H = pcm.toarray() # convert to numpy array
        else:
            self.H = pcm # use the input as is if it is already a numpy array
        n_cols = self.H.shape[1]
        n_rows = self.H.shape[0]
        # construct compact H matrix
        if priors is None:
            if p is None:
                raise ValueError("Either priors or p must be provided.")
            else:
                priors = np.full(n_cols, 2 * p / 3, dtype=np.float32)  # Default priors based on p

        self.priors = np.array(priors)
        self.i_max = i_max
        self.bpd = BpDecoder(
                    pcm, #the parity check matrix
                    error_channel=priors, # the error rate on each bit
                    max_iter=i_max, #the maximum iteration depth for BP
                    bp_method="ms", #BP method.
                    ms_scaling_factor=0,
                    schedule=scheduling)
        self.w_min = w_min
        self.w_max = w_max
        self.n_sample = n_sample
        self.topk = topk
        self.max_procs = max_procs
        if max_procs > 1:
            self.input_queue = mp.Queue()
            self.output_queue = mp.Queue()
            self.solution_found_event = mp.Event()
            self.procs = [mp.Process(target=decode_worker,
                                     args=(self.H, self.H_csc, self.priors, self.i_max, self.input_queue, self.output_queue, self.solution_found_event),
                                     daemon=True)
                          for _ in range(max_procs)]
            for p in self.procs:
                p.start()
        else:
            self.procs = None
        self.synd_serial_num = 0
    
    def __del__(self):
        if self.max_procs > 1:
            for p in self.procs:
                self.input_queue.put(None)  # One sentinel per process
            for p in self.procs:
                p.join(timeout=5)  # Add timeout to prevent hanging
    
    def trial_parallel(self, syndrome, candidates):
        self.synd_serial_num += 1
        flg = False
        self.solution_found_event.clear()

        decoded = np.zeros(self.H.shape[1], dtype=int)
        
        all_combs = []
        for w in range(self.w_min, self.w_max + 1):
            all_combs.extend(sample_n_choose_k(candidates, w, self.n_sample))
        
        if not all_combs:
            return False, decoded

        chunk_size = math.ceil(len(all_combs) / self.max_procs)
        chunks = [all_combs[i:i + chunk_size] for i in range(0, len(all_combs), chunk_size)]

        for chunk in chunks:
            self.input_queue.put((syndrome, chunk, self.synd_serial_num))

        chunks_processed = 0
        while chunks_processed < len(chunks):
            try:
                res_flg, res_decoded, res_serial_num = self.output_queue.get(timeout=1.0)
            except Empty:
                if self.solution_found_event.is_set():
                    # If a solution was found but the queue is empty, we can exit early.
                    break
                continue

            if res_serial_num != self.synd_serial_num:
                continue # Stale result from previous call

            if res_flg:
                self.solution_found_event.set() # Signal other workers to stop
                # Clean up queues and return the successful result
                while not self.input_queue.empty():
                    try:
                        self.input_queue.get_nowait()
                    except Empty:
                        break
                while not self.output_queue.empty():
                    try:
                        self.output_queue.get_nowait()
                    except Empty:
                        break
                return True, res_decoded
            else:
                # This was a "chunk done" signal
                chunks_processed += 1
        
        # Fallback in case no solution is found after all chunks are processed
        return False, decoded


    def trials(self, syndrome, candidates):
        for w in range(self.w_min, self.w_max+1):
            comb = sample_n_choose_k(candidates, w, self.n_sample)
            for test_pos in comb:
                test_pos = np.array(test_pos)
                test_syndrome = self.compute_syndrome_sparse(test_pos)
                new_syndrome = np.remainder(syndrome + test_syndrome, 2)
                decoded = self.bpd.decode(new_syndrome)
                flg = self.bpd.converge
                if flg:
                    decoded[test_pos] = 1 - decoded[test_pos] # Flip the bits in the decoded codeword
                    return flg, decoded
        return False, decoded

    def generate_precompute(self, syndrome):
        decoded = self.bpd.decode(syndrome)
        flg = self.bpd.converge
        if flg:
            return True, syndrome
        else:
            candidates = np.argsort(self.bpd.oscillation_counts)[-self.topk:]
            return False, syndrome, candidates

    def flip_decode(self, syndromes):
        """
        Perform flipping-BP decoding for a batch of syndromes using batch_decode.

        Args:
            syndromes (np.ndarray): Batch of syndrome vectors (shape: [batch_size, num_checks]).
            p (float): Channel error probability.
            topk (int): Number of top bits to consider for flipping.
            w_max (int): Maximum weight of the flipping vector.
            parallel (bool): If True, use parallel decoding.
            n_sample (int): Number of samples to generate for each combination.

        Returns:
            tuple: (success (np.ndarray), decoded_codewords (np.ndarray))
        """
        decoded = self.bpd.decode(syndromes)
        success = self.bpd.converge

        # If all syndromes are successfully decoded, return the results
        if success or self.w_max == 0:
            return success, decoded
        else:
            candidates = np.argsort(self.bpd.oscillation_counts)[-self.topk:]
            if self.max_procs > 1:
                success, decoded = self.trial_parallel(syndromes, candidates)
            else:
                success, decoded = self.trials(syndromes, candidates)
            return success, decoded


    def compute_syndrome_sparse(self, test_positions):
        """Compute syndrome for sparse test vector efficiently"""
        syndrome_cols = self.H_csc[:, test_positions]
        syndrome = np.array(syndrome_cols.sum(axis=1) % 2, dtype=np.uint8)
        return syndrome.flatten()
    
        
class NV_Fake:
    """Generating fake candidates only for time estimation"""
    def __init__(self, pcm, i_max, priors=None, p=None, w_min=0, w_max=0, n_sample=0, topk=0,
        batch_size=1000):
        self.topk = topk
        self.w_min = w_min
        self.w_max = w_max
        self.n_sample = n_sample
        self.H = pcm.toarray() 
        n_cols = self.H.shape[1]
        if priors is None:
            if p is None:
                raise ValueError("Either priors or p must be provided.")
            else:
                priors = np.full(n_cols, 2 * p / 3, dtype=np.float32)  # Default priors based on p
        opts = dict() 
        opts['error_rate_vec'] = priors
        opts['max_iterations']=i_max
        opts['use_osd']=False
        opts['bp_batch_size']=batch_size
        opts['use_sparsity']=True
        chkDenseForNV = np.array(pcm.todense(order='C'))
        self.nvdec = qec.get_decoder('nv-qldpc-decoder', chkDenseForNV, **opts)
        opts_single = dict()
        opts_single['error_rate_vec'] = priors
        opts_single['max_iterations']=i_max
        opts_single['use_osd']=False
        opts_single['use_sparsity']=True
        self.nvdec_single = qec.get_decoder('nv-qldpc-decoder', chkDenseForNV, **opts_single)

    def flip_decode_fake(self, input_data):
        syndrome = input_data[1]
        precompute_success = input_data[0]
        result = self.nvdec_single.decode(syndrome)
        flag = result.converged
        if flag and precompute_success:
            return result
        elif flag != precompute_success: # nvidia decoder can not decode, but CPU BP can, or otherway around
            return None
        else: # nvidia decoder can not decode, and CPU decoder can not decode, use precompute candidates
            candidates = input_data[2]

        for w in range(self.w_min, self.w_max+1):
            comb = sample_n_choose_k(candidates, w, self.n_sample)
            for test_pos in comb:
                test_pos = np.array(test_pos)
                new_syndrome = compute_syndrome_sparse(self.H, test_pos)
                test_syndrome = np.remainder(syndrome + new_syndrome, 2)
                result = self.nvdec.decode(test_syndrome)
                if result.converged:
                    return result
        return result