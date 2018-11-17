from src.util import find_motifes2, stamp, stomp
import copy
import numpy as np
import pandas as pd

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

    def get_dash_figure(self):
        return {
            'data': [self.get_dash_data()],
            'layout': {
                'title': f'{self.name} Visualization'
            }
        }

class mSequence():
    pass

class SequenceList:
    def __init__(self, motives, distance=-1):
        self.motives = [ResultSequence(sequence=m, name=f"Motif at {idx}", start=idx) for m, idx in motives]
        self.distance = distance
        #self.x_column = x_column
    
    def get_all_idx(self):
        pass

    def get_data_array(self):
        data = []
        for m in self.motives:
            data.append(m.get_dash_data())
        return data

    def get_hist(self):
        self.motives.sort(key=lambda x: x.start)
        distances = {}
        for i in range(len(self.motives) - 1):
            dist = self.motives[i+1].start - self.motives[i].start
            if not dist in distances.keys():
                distances[dist] = 1
            else:
                distances[dist] += 1
                

        return{
            'data': [
                {
                    'x': list(distances.keys()),
                    'y': list(distances.values()),
                    'name': 'Histogram',
                    'type': 'bar'
                }
            ],
            'layout': {
                'title': f'Nearest Neighbour Histogram'
            }
        }
    

    def get_table(self):
        result = pd.DataFrame(map(lambda r: [r.start] ))

    def get_dash_figure(self):
        return {
            'data': self.get_data_array(),
            'layout': {
                'title': f'List Visualization'
            }
        }
 

class RawSequence(Sequence):
    def __init__(self, **kwargs):
        self._lag = -1
        self.matrix_profile = None
        self.matrix_index = None
        super(RawSequence, self).__init__(**kwargs)

    def get_matrix_profile(self, m):
        if self._lag != m:
            self._lag = m
            #profile, index = stamp(copy.deepcopy(self.sequence), copy.deepcopy(self.sequence), m)
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
            result.append(SequenceList(m))
        return result

    def get_disorders(self, number_disorders) -> SequenceList: 
        if self.matrix_profile is None:
            raise Exception("You have to calculate the Matrix Profile First")
        result = []
        matrix_profile = self.matrix_profile.sequence
        for _ in range(0, number_disorders):
            max = np.argmax(matrix_profile)
            result.append((self.sequence[max:max + self._lag], max))
            matrix_profile[max - self._lag // 2: max + self._lag //2] = -1
        return SequenceList(result)


class ResultSequence(Sequence):
    def __init__(self, start=-1, **kwargs):
        self.start = int(start)
        super(ResultSequence, self).__init__(**kwargs)
