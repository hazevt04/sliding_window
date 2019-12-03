#!/usr/bin/env python3
import numpy as np
import cupy as cp
import random
import time

# Script for sliding_window
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()        
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print("{}(): {} ms".format(method.__name__, (te-ts)*1000))
            #%r  %2.2f ms' % \
            #      (method.__name__, (te - ts) * 1000)
        return result
    return timed

@timeit
def sliding_window( vals, window_size ):
    num_vals = len(vals)
    num_sums = num_vals - window_size
    sums = np.zeros((num_sums,1))
    
    for i in range( 0, window_size ):
        sums += vals[i:i-window_size]
    
    print("Sums[:10] is {}".format(sums[:10]))
    return sums

@timeit
def sliding_window_alt( vals, window_size ):
    num_vals = len(vals)
    num_sums = num_vals - window_size
    
    sums = np.zeros((num_sums,1))
    sums[0] = np.sum(vals[0:window_size])

    for i in range( 1, num_sums ):
        sums[i] = sums[i-1] - vals[i-1] + vals[i+window_size-1]

    print("alt: Sums[:10] is {}".format(sums[:10]))
    return sums


def sliding_window_cp( vals, window_size ):
    num_vals = len(vals)
    num_sums = num_vals - window_size
    sums = cp.zeros((num_sums,1))
        
    for i in range( 0, window_size ):
        sums += vals[i:i-window_size]
        
    print("Sums[:10] is {}".format(sums[:10]))
    return sums


def sliding_window_alt_cp( vals, window_size ):
    num_vals = len(vals)
    num_sums = num_vals - window_size
    
    sums = cp.zeros((num_sums,1))
    sums[0] = cp.sum(vals[0:window_size])

    for i in range( 1, num_sums ):
        sums[i] = sums[i-1] - vals[i-1] + vals[i+window_size-1]

    #print("alt: Sums[:10] is {}".format(sums[:10]))
    return sums


if __name__ == '__main__':
    window_size = 4000
    num_vals = 1000000
    vals = np.random.random((num_vals,1))
    print("Vals[:10] is {}".format(vals[:10]))
    sums = sliding_window( vals, window_size ) 
    sums_alt = sliding_window_alt( vals, window_size ) 

    if not np.allclose(sums, sums_alt):
        m_index = np.where(not np.isclose(sums,sums_alt))
        print("Mismatch: sums[{}] = {}, sums_alt[{}] = {}".format(m_index, sums[m_index], sums_alt[m_index]))

    gpu_start_time = time.time()
    gpu_vals = cp.array(vals)
    gpu_sums_alt = sliding_window_cp( gpu_vals, window_size ) 
    sums_alt = cp.asnumpy(gpu_sums_alt)
    print("GPU Time: sliding_window_cp(): {} ms".format((time.time() - gpu_start_time)*1000.0))

    if not np.allclose(sums, sums_alt):
        m_index = np.where(not np.isclose(sums,sums_alt))
    gpu_start_time = time.time()
    gpu_vals = cp.array(vals)
    gpu_sums_alt = sliding_window_alt_cp( gpu_vals, window_size ) 
    sums_alt = cp.asnumpy(gpu_sums_alt)
    print("GPU Time: sliding_window_alt_cp(): {} ms".format((time.time() - gpu_start_time)*1000.0))

    if not np.allclose(sums, sums_alt):
        m_index = np.where(not np.isclose(sums,sums_alt))
        print("Mismatch: sums[{}] = {}, gpu_sums_alt[{}] = {}".format(m_index, sums[m_index], sums_alt[m_index]))
