from src.util import find_motifes2, stamp, stomp, find_aligned_motifes
import copy
import numpy as np
import pandas as pd
import dash_core_components as dcc

class Sequence:
    """
    Class which represents an Sequence, and associated informations
    """
    def __init__(self, sequence=None, name=None, x_column=None, y_columns=None):
        self.sequence = sequence
        self.name = name
        self.x_column = x_column
        self.y_columns = y_columns

    def get_dash_data(self):
        x = self.x_column
        if x is None or len(x) == 0:
            x = [i for i in range(len(list(self.sequence.flat)))]
        y = list(self.sequence.flat)
        if self.y_columns is not None and len(self.y_columns) != 0:
            y = self.y_columns
        return {'y': y, 'x': x, 'type':'scatter', 'mode':'line', 'name': self.name }

    def get_dash_figure(self, id):
        return dcc.Graph(
            id = id,
            figure = {
            'data': [self.get_dash_data()],
            'layout': {
                'title': f'{self.name} Visualization',
                'xaxis':{
                    'title':'Time',
                },
                'yaxis':{
                    'title':'Value'
                },
                'font':{
                    'size':18,
                }
            },
        })

class mSequence():
    pass

class SequenceList:
    def __init__(self, motives, distance=-1, x_column=None):
        name_mapping = lambda x: x
        if not x_column is None:
            name_mapping = lambda x: x_column[x]
        self.motives = [ResultSequence(sequence=m, name=f"{name_mapping(idx)}", start=idx, time=name_mapping(idx)) for m, idx in motives]
        self.distance = distance
        self.x_column = x_column
    
    def get_all_idx(self):
        pass

    def get_data_array(self):
        data = []
        for m in self.motives:
            data.append(m.get_dash_data())
        return data

    def get_weekday_hist(self, id):
        weekday = {0:0,1:0,2:0,3:0,4:0,5:0,6:0}
        for d in self.motives:
            weekday[d.time.weekday()] += 1
        return dcc.Graph(
            id=id,
            figure = {
                'data': [
                    {
                        'x': list(weekday.keys()),
                        'y': list(weekday.values()),
                        'name': 'Histogram',
                        'type': 'histogram',
                        'histfunc':'sum',
                        'xbins':{
                            'start':0,
                            'size':1
                        } 
                    }
                ],
                'layout': {
                    'title': f'Weekday Histogram',
                    'xaxis':{
                        'title':'Weekday',
                    },
                    'yaxis':{
                        'title':'Count'
                    },
                    
                    'font':{
                        'size':18,
                    }
                },
            })
    
    def get_month_hist(self, id):
        month = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0}
        for d in self.motives:
            month[d.time.month] += 1
        return dcc.Graph(
            id=id,
            figure = {
                'data': [
                    {
                        'x': list(month.keys()),
                        'y': list(month.values()),
                        'name': 'Histogram',
                        'type': 'histogram',
                        'histfunc':'sum',
                        'xbins':{
                            'start':0,
                            'size':1
                        } 
                    }
                ],
                'layout': {
                    'title': f'Weekday Histogram',
                    'xaxis':{
                        'title':'Month',
                    },
                    'yaxis':{
                        'title':'Count'
                    },
                    'font':{
                        'size':18,
                    }
                }
            })
    
    def get_hist(self, id):
        #self.motives.sort(key=lambda x: x.start)
        distances = {}
        for i in range(len(self.motives) - 1):
            for j in range(i + 1, len(self.motives)):
                dist = np.abs(self.motives[j].start - self.motives[i].start)
                if not dist in distances.keys():
                    distances[dist] = 1
                else:
                    distances[dist] += 1
        return dcc.Graph(
        id=id,
        figure={
            'data': [
                {
                    'x': list(distances.keys()),
                    'y': list(distances.values()),
                    'name': 'Histogram',
                    'type': 'histogram',
                    'histfunc':'sum',
                    'xbins':{
                        'start':0,
                        'size':48
                    } 
                }
            ],
            'layout': {
                'title': f'Nearest Neighbour Histogram',
                'xaxis':{
                        'title':'distance',
                    },
                    'yaxis':{
                        'title':'count'
                    },
                'font':{
                    'size':18,
                }
            }
        })

    def get_table(self):
        result = pd.DataFrame(map(lambda r: [r.start] ))

    def get_dash_figure(self, id):
        return dcc.Graph(
            id = id,
            figure = {
            'data': self.get_data_array(),
            'layout': {
                'title': f'Pattern',
                'xaxis':{
                    'title':'Time',
                },
                'yaxis':{
                    'title':'Value'
                },
                'font':{
                    'size':18,
                }
            }
        })
 

class RawSequence(Sequence):
    def __init__(self, **kwargs):
        self._lag = -1
        self.matrix_profile = None
        self.matrix_index = None
        super(RawSequence, self).__init__(**kwargs)

    def get_matrix_profile(self, m):
        if self._lag != m:
            self._lag = m
            profile, index = stomp(self.sequence,m)
            self.matrix_profile = Sequence(profile, "Matrix Profile", x_column=self.x_column)
            self.matrix_index = Sequence(index, "Matrix Profile Index", x_column=self.x_column, y_columns=list(map(lambda x: self.x_column[x], index.flat)))
        return self.matrix_profile

    def get_motives(self, m):
        if self._lag != m:
            self._lag = m
            self.get_matrix_profile(m)
        motives = find_motifes2(self.sequence, self.matrix_profile.sequence, self.matrix_index.sequence, m)
        result = []
        for m in motives:
            result.append(SequenceList(m, x_column=self.x_column))
        return result

    def get_av_motives(self, av, m):
        mp = self.matrix_profile.sequence + (1 - av) * np.max(self.matrix_profile.sequence[self.matrix_profile.sequence < np.inf])
        motives = find_motifes2(self.sequence, mp, self.matrix_index.sequence, m)
        result = []
        for m in motives:
            result.append(SequenceList(m, x_column=self.x_column))
        return result

    def get_aligned_motives(self,m, start=0, offset=1):
        motives = find_aligned_motifes(self.sequence, self.matrix_profile.sequence, self.matrix_index.sequence, m, start=start, offset=offset)
        result = []
        for m in motives:
            result.append(SequenceList(m, x_column=self.x_column))
        return result


    def get_disorders(self, number_disorders) -> SequenceList: 
        if self.matrix_profile is None:
            raise Exception("You have to calculate the Matrix Profile First")
        result = []
        matrix_profile = self.matrix_profile.sequence.copy()
        for _ in range(0, number_disorders):
            max = np.argmax(matrix_profile)
            result.append((self.sequence[max:max + self._lag], max))
            matrix_profile[max - self._lag // 2: max + self._lag //2] = -1
        return SequenceList(result, x_column=self.x_column)


class ResultSequence(Sequence):
    def __init__(self, start=-1, time=None, **kwargs):
        self.start = int(start)
        self.time = time
        super(ResultSequence, self).__init__(**kwargs)
