import numpy as np
import scipy.signal as sci
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from numba import njit, prange

#@njit(fastmath=True)
def slidingDotProduct(Q, T):
    """
    ATTENTION: Standardizise your time series before using this variant. If some values are to high,
    an overflow is possible and no warning will be printed
    """
    #T.resize(T.shape[0])
    #Q.resize(Q.shape[0])
    m = len(Q)
    n = len(T)
    return sci.fftconvolve(Q[::-1], T)[m - 1:n]

def timeseries_mean_stddev(T : np.ndarray, lag):
    """
    :return: Array of mean and array of stddev
    """
    n = len(T)
    t_padded = np.concatenate((np.zeros((1, )), T, np.zeros((lag, ))))
    t_cum = np.cumsum(t_padded)
    t2_cum = np.cumsum(t_padded * t_padded)
    t_sum = t_cum[lag:n + 1] - t_cum[:n - lag + 1] 
    t2_sum = t2_cum[lag:n + 1] - t2_cum[:n  - lag + 1] 
    t_mu = t_sum  / lag
    t_sig = np.sqrt(np.abs((t2_sum / lag) - (t_mu * t_mu)))
    return t_mu, t_sig

@njit(parallel=True)
def calculateDistanceProfile(Q, T, QT, T_mean, T_dev, mean_q, stddev_q) -> np.ndarray: 
    m = len(Q)
    var1 = QT - m * mean_q * T_mean
    std = m * T_dev * stddev_q
    return np.sqrt(2 * m * (1 - var1 / std))

def mass(Q, T, T_mean, T_dev, idx):
    #Q_s = StandardScaler().fit_transform(Q)
    QT = slidingDotProduct(Q, T)
    mean_q = np.mean(Q)
    stddev_q =np.std(Q)
    m = len(Q)
    var = QT - m * mean_q * T_mean
    std = m * T_dev * stddev_q
    return np.sqrt(np.abs(2 * m * (1 - var / std)))


def stamp(Ta, Tb, m, max_iter=20000):
    nb = len(Tb) - m + 1
    if max_iter > 0:
        order = np.random.random_integers(0, high=nb - 1, size=max_iter)
        order = set(order)
    else:
        pass
    order = range(nb)
    pab = np.array([np.inf for i in range(nb)])
    iab = np.zeros(nb)
    t_mean, t_sig = timeseries_mean_stddev(Ta, m)
    exc_zone_len = m // 2
    for idx in tqdm(order):
        D = mass(Tb[idx: idx + m], Ta, t_mean, t_sig, idx)
        _set_exclusion_zone(idx, exc_zone_len, nb, D)
        iab[pab > D] = idx
        pab[pab > D] = D[pab > D]
        
    return pab, iab

def m_stamp(Ta, m, max_iter=5000):
    dimension = Ta.shape[1]
    n = Ta.shape[0]
    if max_iter > 0:
        order = np.random.random_integers(0, high=n-m +1, size=max_iter)
        order = set(order)
    else:
        order = range(n-m)
    matrix_profile = np.full((dimension, n - m + 1), np.inf)
    matrix_index = np.full((dimension, n - m + 1), -1)
    #dim_index = np.full((dimension, n - m + 1), -1)
    for i in tqdm(order):
        distance_profile = np.zeros((dimension, n - m + 1))
        for j in range(dimension):
            Q = Ta[i : i + m, j]
            T = Ta[:,j]
            t_mean, t_std = timeseries_mean_stddev(T, m)
            distance_profile[j] = mass(Q, T, t_mean, t_std, j)
        distance_profile_idx = np.argsort(distance_profile, axis=0)
        distance_profile = np.sort(distance_profile, axis=0)
        distance_profile_cummulated = np.zeros(n - m + 1)
        for j in range(dimension):
            distance_profile_cummulated += distance_profile[j,:]
            d_ = distance_profile_cummulated / (j + 1)
            exc_zone_len = m // 2
            exc_zone_start = 0 if i < exc_zone_len else int(i - exc_zone_len)
            exc_zone_end = n if i > n - exc_zone_len else int(i + exc_zone_len)
            d_[exc_zone_start : exc_zone_end] = np.inf
            temp = distance_profile_idx[:1+j, matrix_profile[j, :] > d_]
            #dim_index[j, matrix_profile[j, :] > d_] =  list(tuple(temp[k,l] for k in range(j+1)) for l in range(temp.shape[1]))

            matrix_index[j, matrix_profile[j, :] > d_] = i
            matrix_profile[j, matrix_profile[j, :] > d_] = d_[matrix_profile[j, :] > d_]
    return matrix_profile, matrix_index#, dim_index
  

def m_stomp(Ta, m):
    dimension = Ta.shape[0]
    n = Ta.shape[1]
    matrix_profile = np.full((dimension, n - m), np.inf)
    for i in range(n - m):
        distance_profile = np.zeros((dimension, n - m))
        #TODO: Neue Variante schreiben, in der die folgende Schleife so ersetzt wird, dass 
            # STOMP statt STAMP verwendet wird.
        for j in range(dimension):
            pass
        distance_profile_idx = np.argsort(distance_profile, axis=0)
        distance_profile = np.sort(distance_profile, axis=0)
        distance_profile_cummulated = np.zeros(n - m)
        for j in range(dimension):
            distance_profile_cummulated += distance_profile[j,:]
            d_ = distance_profile_cummulated / j
            exc_zone_len = np.round(m * 0.5)
            exc_zone_start = 0 if j < exc_zone_len else int(j - exc_zone_len)
            exc_zone_end = n if j > n - exc_zone_len else int(j + exc_zone_len)
            d_[exc_zone_start : exc_zone_end] = np.inf
            matrix_profile[matrix_profile[j] > d_] = d_[matrix_profile > d_]
    return matrix_profile

def stomp(T:np.ndarray, m):
    n = len(T)
    l = n - m 
    mean, dev = timeseries_mean_stddev(T, m)
    QT = np.array(slidingDotProduct(T[0:m], T))
    QT_first = QT.copy()
    D = calculateDistanceProfile(T[0:m], T, QT, mean, dev, mean[0], dev[0])
    P = D
    P[0:m//8] = [np.inf for i in range(m//8)]
    I = np.zeros(D.shape)
    T_t = T[:l+1]
    T_m = T[m -1:]
    #T_m = np.concatenate((np.zeros((1)), T[m:]))[:l+1]
    for i in tqdm(range(1, l)):
        a = np.isnan(QT)
        if np.any(a):
            print("NAN ALARM")
            print(np.argsort(a)[-1])
        QT, P, I = faster(QT, l, T_t, T, i, n, m, QT_first, mean[:l+1], dev[:l +1], I, P)
    return P, I


#ATTENTION DO NOT USE parallel=True, in that case the result becomes wrong
@njit()
def faster(QT, l, T_t, T, i, n, m, QT_first, mean, dev, I, P):
    QT[1:] = QT[0:-1] - T[0:l] * T[i-1] + (T[m:n] * T[i + m - 1])
    QT[0] = QT_first[i]
    D = calculateDistanceProfile(T[i:i+m], T, QT, mean, dev, mean[i], dev[i])
    _set_exclusion_zone(i, m // 2, l, D)

    I[P > D] = i
    P[P > D] = D[P > D]
    return QT, P, I

def find_motifes2(series, matrix_profile:np.ndarray, matrix_index:np.ndarray, m, num_motives = 16):
    result = []
    exc_zone_len = np.round(2 * m)
    t_mean, t_std = timeseries_mean_stddev(series, m)
    nb = len(series) - m + 1
    mp = matrix_profile.copy()
    for _ in range(num_motives):
        motives = []

        idx = np.argmin(mp)
        dist_len = mp[idx]
        dist_len = dist_len * dist_len
        print("============", dist_len)
        motives.append((series[idx : idx + m], idx))

        dist_profile = calculateDistanceProfile(series[idx:idx + m], series, slidingDotProduct(series[idx:idx + m], series),t_mean, t_std, np.mean(series[idx:idx + m]), np.std(series[idx:idx + m]))
        _set_exclusion_zone(idx, exc_zone_len, nb, dist_profile)
        _set_exclusion_zone(idx, exc_zone_len, nb, mp)

        idx = int(matrix_index[idx])
        motives.append((series[idx : idx + m], idx))
        _set_exclusion_zone(idx, exc_zone_len, nb, dist_profile)
        _set_exclusion_zone(idx, exc_zone_len, nb, mp)
        if np.isnan(dist_len):
            print(len(motives))
            result.append(motives)
            continue
        dist_profile_idx = dist_profile.argsort()
        dist_profile.sort()

        for i, idx in enumerate(dist_profile_idx):
            if dist_profile[i] > dist_len:
                break
            if mp[idx] != np.inf:
                motives.append((series[idx: idx + m],idx))
                _set_exclusion_zone(idx, exc_zone_len, nb, mp)
        print(len(motives))
        result.append(motives)
    return result  

@njit()
def _set_exclusion_zone(idx, exc_zone_len, nb, mp):
    exc_zone_start = 0 if idx < exc_zone_len else int(idx - exc_zone_len / 2)
    exc_zone_end = nb if idx > nb - exc_zone_len else int(idx + exc_zone_len / 2)
    mp[exc_zone_start : exc_zone_end] = [np.inf for i in range(exc_zone_end - exc_zone_start)]

