import dash
#import dash-table
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.sequence import Sequence, RawSequence,SequenceList
from src.util import m_stamp

series_orig = pd.read_csv("test_set\Electricity_CN_Subset.csv")[["Bldg.124", "Time"]]
series = series_orig.drop("Time", axis=1)
#std = StandardScaler()
#series = std.fit_transform(series)
series = np.array(list(series.values.flat))

r_series = RawSequence(sequence=series, name="Internet Traffic", x_column=series_orig["Time"])

mp = pd.read_csv("mps\matrix_profileBldg.124.csv")
mi = pd.read_csv("mps\matrix_indexBldg.124.csv")
r_series.matrix_profile = Sequence(mp['0'].values, "Matrix Profile", x_column=r_series.x_column)
print("loaded matrix_profile")
r_series.matrix_index = Sequence(mi['0'].values, "Matrix Profile Index", x_column=r_series.x_column, y_columns=list(map(lambda x: r_series.x_column[x], mi['0'].values)))
print("loaded matrix index")
motives = r_series.get_motives(144)
disorders = r_series.get_disorders(15)

app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='Matrix Profile Dash'),

    dcc.Graph(
        id='series',
        figure=r_series.get_dash_figure()
    ),

    dcc.Graph(
        id='profile',
        figure=r_series.matrix_profile.get_dash_figure()
    ),

    dcc.Graph(
        id='index',
        figure=r_series.matrix_index.get_dash_figure()
    ),
    dcc.Graph(
        id='motive',
        figure=motives[0].get_dash_figure()
    ),
    #dcc.Graph(
    #    id='hist',
    #    figure=motives[0].get_hist()
    #),
   # dash_table.DataTable(
   #     id='tablem1',
   #     columns=[{"name": i, "id": i} for i in df.columns],
   #     data=motives[0].get_table().to_dict("rows"),
   # ),
    dcc.Graph(
        id='motive2',
        figure=motives[1].get_dash_figure()
    ),
    dcc.Graph(
        id='motive3',
        figure=motives[2].get_dash_figure()
    ),
    dcc.Graph(
        id='motive4',
        figure=motives[3].get_dash_figure()
    ),
    dcc.Graph(
        id='motive5',
        figure=motives[4].get_dash_figure()
    ),
    dcc.Graph(
        id='motive6',
        figure=motives[5].get_dash_figure()
    ),
    dcc.Graph(
        id='motive7',
        figure=motives[6].get_dash_figure()
    ),
        dcc.Graph(
        id='motive8',
        figure=motives[7].get_dash_figure()
    ),
    dcc.Graph(
        id='hist8',
        figure=motives[7].get_hist()
    ),    dcc.Graph(
        id='motive9',
        figure=motives[8].get_dash_figure()
    ),
        dcc.Graph(
        id='hist9',
        figure=motives[8].get_hist()
    ),
    dcc.Graph(
        id='motive10',
        figure=motives[9].get_dash_figure()
    ),
        dcc.Graph(
        id='hist10',
        figure=motives[9].get_hist()
    ),
    dcc.Graph(
        id='motive11',
        figure=motives[10].get_dash_figure()
    ),
        dcc.Graph(
        id='hist11',
        figure=motives[10].get_hist()
    ),
    dcc.Graph(
        id='motive12',
        figure=motives[11].get_dash_figure()
    ),
        dcc.Graph(
        id='hist12',
        figure=motives[11].get_hist()
    ),
        dcc.Graph(
        id='motive13',
        figure=motives[12].get_dash_figure()
    ),
        dcc.Graph(
        id='hist13',
        figure=motives[12].get_hist()
    ),
    dcc.Graph(
        id='motive14',
        figure=motives[13].get_dash_figure()
    ),
        dcc.Graph(
        id='hist14',
        figure=motives[13].get_hist()
    ),
    dcc.Graph(
        id='motive15',
        figure=motives[14].get_dash_figure()
    ),
        dcc.Graph(
        id='hist15',
        figure=motives[14].get_hist()
    ),
    dcc.Graph(
        id='motive16',
        figure=motives[15].get_dash_figure()
    ),
    dcc.Graph(
        id='hist16',
        figure=motives[15].get_hist()
    ),
    dcc.Graph(
        id='disorders',
        figure=disorders.get_dash_figure()
    )
])

if __name__ == '__main__':
    
    app.run_server(debug=True)