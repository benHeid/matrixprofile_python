import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.sequence import Sequence, RawSequence,SequenceList
from src.util import m_stamp

#Adapt this variables
COLUMN ="Column"
TIME_COLUMN = "Time"
TIME_SERIES_PATH = "path\data.csv"
MATRIXPROFILE_PATH = "matrixprofile.csv"
MATRIXPROFILE_INDEX_PATH = "matrixprofileindex.csv"
SEQUENCE_LENGTH = 96
NUMBER_DISORDERS = 15
NUMBER_MOTIFES = 15

series_orig = pd.read_csv(TIME_SERIES_PATH)[[COLUMN, TIME_COLUMN]]
series = series_orig.drop("Time", axis=1)
series = np.array(list(series.values.flat))

r_series = RawSequence(sequence=series, name="Name", x_column=pd.to_datetime(series_orig[TIME_COLUMN]))

mp = pd.read_csv(MATRIXPROFILE_PATH)
mi = pd.read_csv(MATRIXPROFILE_INDEX_PATH)
r_series.matrix_profile = Sequence(mp['0'].values, "Matrix Profile", x_column=r_series.x_column)
print("loaded matrix_profile")
r_series.matrix_index = Sequence(mi['0'].values, "Matrix Profile Index", x_column=r_series.x_column, y_columns=list(map(lambda x: r_series.x_column[x], mi['0'].values)))
print("loaded matrix index")
motives = r_series.get_motives(SEQUENCE_LENGTH)
disorders = r_series.get_disorders(NUMBER_DISORDERS)

app = dash.Dash(__name__)
children = [html.H1(children='Matrix Profile Dash'),
    r_series.get_dash_figure(id="series"),
    r_series.matrix_profile.get_dash_figure(id="profile"),
    r_series.matrix_index.get_dash_figure(id="index")]

for i in range(NUMBER_MOTIFES):
    children.extend(
        [motives[i].get_dash_figure(id=f"motive{i}"),
        motives[i].get_hist(id=f"hist{i}"),
        motives[i].get_weekday_hist(id=f"weekday{i}"),
        motives[i].get_month_hist(id=f"month{i}")])

children.append(disorders.get_dash_figure(id="disorders"))

app.layout = html.Div(children=children)

if __name__ == '__main__':
    
    app.run_server(debug=True)