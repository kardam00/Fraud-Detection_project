#Importing the required modules
import numpy as np
from scipy.stats import mode
import pandas as pd

import time
from typing import Iterator, List, Collection, Callable

from tqdm import tqdm

PREDICTION_FUNCTION_HINT = Callable[
    [Collection, Collection[Collection], Collection[bool]], bool
]

def load_data(df_name: str) -> pd.DataFrame:

    if df_name == 'cars':
        return load_cars()
    
    elif df_name == 'seismic-bumps':
        return load_seismic()
    
    elif df_name == 'ThoracicSurgery':
        return  load_ThoracicSurgery()
    
    elif df_name == 'Breast_Cancer':
        return load_Breast_Cancer()

    raise ValueError(f'Unknown dataset name: {df_name}')

def load_cars() -> pd.DataFrame:
    column_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety','class'
    ]

    url = 'https://raw.githubusercontent.com/kardam00/Lazy_FCA/main/Datasets/car.csv'
    #df = pd.DataFrame(r.text)
    df = pd.read_csv(url)
    df.rename(columns = {'vhigh':'buying', 'vhigh.1':'maint',
                              '2':'doors', '2.1':'persons', 'small':'lug_boot','low':'safety','unacc':'class'},
              inplace = True)
    df['class'] = [x == 'good' or x == 'vgood' for x in df['class']]
    return df

def load_seismic() -> pd.DataFrame:

    url = 'https://raw.githubusercontent.com/kardam00/Lazy_FCA/main/Datasets/seismic-bumps.csv'
    df = pd.read_csv(url)
    df.columns = ['seismic', 'seismoacoustic', 'shift', 'genergy', 'gpuls', 'gdenergy', 'gdpuls', 
                  'ghazard', 'nbumps', 'nbumps2', 'nbumps3', 'nbumps4', 'nbumps5', 'nbumps6', 'nbumps7',
                  'nbumps89', 'energy', 'maxenergy', 'class']
    df['class'] = [x == 1 for x in df['class']]
    return df

def load_ThoracicSurgery() -> pd.DataFrame:

    url = 'https://raw.githubusercontent.com/kardam00/Lazy_FCA/main/Datasets/ThoracicSurgery.csv'
    df = pd.read_csv(url)
    df.columns = ['DGN','PRE4','PRE5','PRE6','PRE7','PRE8','PRE9','PRE10','PRE11',
                 'PRE14','PRE17','PRE19','PRE25','PRE30','PRE32','AGE', 'class']
    df['class'] = [x == 'T' for x in df['class']]
    return df

def load_Breast_Cancer() -> pd.DataFrame:

    url = 'https://raw.githubusercontent.com/kardam00/Lazy_FCA/main/Datasets/Breast_Cancer.csv'
    df = pd.read_csv(url)
    
    df.columns = ['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA', 'Leptin', 'Adiponectin',
                  'Resistin', 'MCP.1', 'class']
    df['class'] = [x == 1 for x in df['class']]
    return df

def binarize_X(X: pd.DataFrame) -> 'pd.DataFrame[bool]':
    dummies = [pd.get_dummies(X[f], prefix=f, prefix_sep=': ') for f in X.columns]
    X_bin = pd.concat(dummies, axis=1).astype(bool)
    
    return X_bin

#Euclidean Distance
def eucledian(p1,p2):

    dist = np.sqrt(np.sum(np.power((p1-p2),2)))
    return dist
 
#Function to calculate KNN
def predict(x_train, y , x_input, k=5) ->bool:
    op_labels = []
         
    #Array to store distances
    point_dist = []
    #Loop through each training Data
    for j in range(0, len(x_train)):
        distances = eucledian(np.array(x_train.iloc[[j]]) , np.array(x_input))
        #Calculating the distance
        point_dist.append(distances)
            
    point_dist = np.array(point_dist) 
         
    #Sorting the array while preserving the index
    #Keeping the first K datapoints
    dist = np.argsort(point_dist)[:k] 
    
    labels = np.array(y)[dist]
         
    #Majority voting
    lab = mode(labels) 
    lab = lab.mode[0]
    op_labels.append(lab)
 
    return op_labels

def predict_array(X: List[set], Y: List[bool],n_train: int, 
                  update_train: bool = True, use_tqdm: bool = False,
                  predict_func: PREDICTION_FUNCTION_HINT = predict) -> Iterator[bool]:

    for j in tqdm(
        range(n_train,len(X)),
        initial=n_train, total=len(X),
        desc='Predicting step by step',
        disable=not use_tqdm,
    ):
        i = j - n_train
        x = X.iloc[[j]]
        n_trains = n_train + i if update_train else n_train
        yield predict_func(X[:n_trains], Y[:n_trains], x) 
    
def apply_stopwatch(iterator: Iterator):
    outputs = []
    times = []

    t_start = time.time()
    for out in iterator:
        dt = time.time() - t_start
        outputs.append(out)
        times.append(dt)
        t_start = time.time()

    return outputs, times