import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import StandardScaler
from src.sequence import Sequence, RawSequence
from src.util import *
s_ = pd.read_csv("test_set\Electricity_CN_Subset.csv")

#for s in ["Bldg.124", "Bldg.142","Bldg.243","Bldg.255","Bldg.329"]:
for s in ["Bldg.142","Bldg.243","Bldg.255","Bldg.329"]:
    series_orig = s_[[s, "Time"]]

    series = series_orig.drop("Time", axis=1)
    std = StandardScaler()
    series = std.fit_transform(series)
    series = np.array(series.flat)

    r_series = RawSequence(sequence=series, name="Internet Traffic", x_column=series_orig["Time"])

    mp = r_series.get_matrix_profile(96)
    pd.DataFrame(mp.sequence).to_csv(f"mps_96\\matrix_profile{s}.csv")
    pd.DataFrame(r_series.matrix_index.sequence).to_csv(f"mps_96\\matrix_index{s}.csv")

series = s_.drop("Time", axis=1)
std = StandardScaler()
series = std.fit_transform(series)
t, d = m_stamp(series, 96)
pd.DataFrame(t).to_csv("mps_96\\all.csv")
pd.DataFrame(d).to_csv("mps_96\\all_index.csv")
