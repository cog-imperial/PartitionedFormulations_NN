"""
This file contains implementations of the partitioning strategies described in
Section 3.4 of the manuscript. All functions have the same input/output structure:
    
    Parameters:
        m (array): array of weights to be used in sorting
        N (int): number of partitions to create
    
    Returns:
        partitions (list): list of arrays corresponding to partitioned indices
"""
import numpy as np

def getEqualSize(m, N):
    """Implementation of 'Equal Size' strategy"""
    return np.array_split(np.argsort(m), N)

def getEqualRange(m, N):
    """Implementation of 'Equal Range' strategy"""
    if N < 3:
        print('Equal range strategy requires N > 2; equal size partitions used instead.')
        return np.array_split(np.argsort(m), N)
    else:
        # bins of equal range w/ quantiles for outliers
        quantile=0.05
        [m_low_thresh, m_high_thresh] = np.quantile(m,[quantile, 1-quantile])

        bin_thresh = [min(m), m_low_thresh]
        bin_size = abs( (m_high_thresh - m_low_thresh) / (N-2) )

        for thresh in range(N-3):
            bin_thresh.append( m_low_thresh + (thresh+1)*bin_size )
        
        bin_thresh.extend( [m_high_thresh, max(m)] )
        bin_dist = np.digitize(m,bin_thresh[:-1])

        # init res dict
        temp_res = {}
        temp_weights = {}
        for n in range(N):
            temp_res[n] = []
            temp_weights[n] = []

        # assign samples to bins based on bin_dist
        for idx,n_bin in enumerate(bin_dist):
            temp_res[n_bin-1].append(idx)
            temp_weights[n_bin-1].append(m[idx])

        return [np.asarray(temp_res[key]) for key in temp_res.keys()]

def getRandom(m, N):
    """Implementation of 'Random' partition strategy"""
    partitions = {}
    # init partitions
    for n in range(N):
        partitions[n] = []

    for i in range(len(m)):
        partitions[np.random.randint(N)].append(i)

    return [np.asarray(partitions[key]) for key in partitions.keys()]

def getUnevenMagnitudes(m, N):
    """Implementation of 'Unequal Magnitudes' partition strategy"""
    sortedInd = np.argsort(-np.abs(m))
    index = np.multiply(np.ones((int(np.ceil(len(m)/N)), N)), range(N))
    snake = index.copy()
    snake[1::2] = index[1::2,::-1]
    snake = snake.ravel()[0: len(m)]
    partitions = []
    for i in range(N):
        partitions.append(sortedInd[np.where(snake == i)])
    return partitions
    
