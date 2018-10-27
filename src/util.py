import numpy as np

def slidingDotProduct(Q, T):
    #T = T.get_values()
    T.resize(T.shape[0])
    #Q = Q.get_values()
    Q.resize(Q.shape[0])
    n = len(T)
    m = len(Q)
    Q_reversed = Q[::-1]
    T_fft = np.fft.fft(T, 2*n)
    Q_fft = np.fft.fft(Q_reversed, 2*n)
    QT = np.fft.ifft(T_fft * Q_fft)
    return np.real(QT)[0:n + m - 1]

def timeseries_mean_stddev(T : np.ndarray, lag):
    """
    :return: Array of mean and array of stddev
    """
    n = len(T)
    t_padded = np.concatenate((np.zeros((1, 1)), T, np.zeros((lag, 1))))
    t_cum = np.cumsum(t_padded)
    t2_cum = np.cumsum(t_padded * t_padded)
    t_sum = t_cum[lag:n + 1] - t_cum[:n + 1 - lag] 
    t2_sum = t2_cum[lag:n + 1] - t2_cum[:n + 1 - lag] 
    t_mu = t_sum  / lag
    t_sig = np.sqrt((t2_sum / lag) - (t_mu * t_mu))
    return t_mu, t_sig

def calculateDistanceProfile(Q, T, QT, T_mean, T_dev, mean_q, stddev_q):
    m = len(Q)
    var1 = QT[m-1:len(T)] - m * mean_q * T_mean
    std = m * T_dev * stddev_q
    return np.sqrt(np.abs(2* m * (1 - var1 / std)))

def mass(Q, T, T_mean, T_dev):
    QT = slidingDotProduct(Q, T)
    mean_q = np.mean(Q)
    stddev_q = np.std(Q)
    return calculateDistanceProfile(Q, T, QT, T_mean, T_dev, mean_q, stddev_q)

def stamp(Ta, Tb, m):
    nb = len(Tb) - m + 1
    pab = np.array([np.inf for i in range(nb)])
    iab = np.zeros(nb)
    t_mean, t_sig = timeseries_mean_stddev(Ta, m)
    for idx in range(nb):
        D = mass(Tb[idx: idx + m], Ta, t_mean, t_sig)
        iab[pab > D] = idx
        pab[pab > D] = D[pab > D]
    return pab, iab

