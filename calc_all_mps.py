import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import StandardScaler
from src.sequence import Sequence, RawSequence
from src.util import *


#Adapt this variables
COLUMN ="Column"
TIME_COLUMN = "Time"
TIME_SERIES_PATH = "path\data.csv"
MATRIXPROFILE_PATH = "matrixprofile.csv"
MATRIXPROFILE_INDEX_PATH = "matrixprofileindex.csv""
SEQUENCE_LENGTH = 96

s_ = pd.read_csv(TIME_SERIES_PATH)

for s in [COLUMN]:
    series_orig = s_[[s, TIME_COLUMN]]

    series = series_orig.drop(TIME_COLUMN, axis=1)
    std = StandardScaler()
    series = std.fit_transform(series)
    series = np.array(series.flat)

    r_series = RawSequence(sequence=series, name="Name", x_column=series_orig[TIME_COLUMN])

    mp = r_series.get_matrix_profile(SEQUENCE_LENGTH)
    pd.DataFrame(mp.sequence).to_csv(MATRIXPROFILE_PATH)
    pd.DataFrame(r_series.matrix_index.sequence).to_csv(MATRIXPROFILE_INDEX_PATH)
