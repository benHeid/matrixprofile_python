import numpy as np

def slidingDotProduct(Q, T):
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
    t_padded = np.concatenate(([0], T, np.zeros(lag)))
    t_cum = np.cumsum(t_padded)
    t2_cum = np.cumsum(t_padded * t_padded)
    t_sum = t_cum[lag:n + 1] - t_cum[:n + 1 - lag] 
    t2_sum = t2_cum[lag:n + 1] - t2_cum[:n + 1 - lag] 
    t_mu = t_sum  / lag
    t_sig = np.sqrt((t2_sum / lag) - (t_mu * t_mu))
    return t_mu, t_sig

def mass(Q, T):
    QT = slidingDotProduct(Q, T)
    #mean_q, stddev_q, mean_T, sigma_T = computeMeanStd()

def computeMeanStd():
    pass