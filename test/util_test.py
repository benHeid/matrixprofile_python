from pytest import fixture
from src.util import slidingDotProduct, timeseries_mean_stddev, calculateDistanceProfile
import numpy as np
from sklearn.preprocessing import StandardScaler

@fixture
def getArray():
    return 


class TestUtil(object):
    def test_slide_product(self):
        a = np.array([1,2,3,1,2,3])
        b = np.array([4,5,6])
        ab = slidingDotProduct(b,a)
        t = np.convolve(a,b)
        assert np.all(np.round(ab).astype(np.integer) == np.array([6, 17, 32, 29, 29, 32, 23, 12]))

    def test_timeseries_mean_stddev(self):
        a = np.array([[1],[2],[3],[1],[2],[3],[1],[2],[3]])
        mean, stddev = timeseries_mean_stddev(a, 3)
        assert len(mean) == 7
        assert len(stddev) == 7
        assert np.all(np.round(mean) == np.array([2.0,2.0,2.0,2.0,2.0,2.0,2.0]))
        assert np.all(np.round(stddev) == np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0])) 

    #def test_calculateDistanceProfile(self):
    #    scaler = StandardScaler()
    #    a = np.array([[1.0],[2.0],[1000.0]])
    #   # a = scaler.fit_transform(a)
    #    b = a[0:2]
#
    #    m, s = timeseries_mean_stddev(a, 2)
    #    dist = calculateDistanceProfile(b, a, slidingDotProduct(b,a), m, s, np.mean(b), np.std(b))
    #    assert len(dist) == 2
    #    assert np.all(dist == [0,np.sqrt(2)])

