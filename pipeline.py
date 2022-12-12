import time
from typing import Iterator, List, Collection, Callable

import pandas as pd
from tqdm import tqdm

PREDICTION_FUNCTION_HINT = Callable[
    [Collection, Collection[Collection], Collection[bool]], bool
]

def load_data(df_name: str) -> pd.DataFrame:
    """Generalized function to load datasets in the form of pandas.DataFrame"""
    if df_name == 'cars':
        return load_cars()
    
    elif df_name == 'seismic-bumps':
        return load_seismic()
    
    elif df_name == 'ThoraricSurgery':
        return  load_ThoraricSurgery()
    
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

def load_ThoraricSurgery() -> pd.DataFrame:

    url = 'https://raw.githubusercontent.com/kardam00/Lazy_FCA/main/Datasets/ThoraricSurgery.csv'
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
    """Scale values from X into pandas.DataFrame of binary values"""
    
    dummies = [pd.get_dummies(X[f], prefix=f, prefix_sep=': ') for f in X.columns]
    X_bin = pd.concat(dummies, axis=1).astype(bool)
    
    return X_bin


def predict_with_generators(
        x: set, X_train: List[set], Y_train: List[bool],
        min_cardinality: int = 1
) -> bool:
    X_pos = [x_train for x_train, y in zip(X_train, Y_train) if y]
    X_neg = [x_train for x_train, y in zip(X_train, Y_train) if not y]

    n_counters_pos = 0  # number of counter examples for positive intersections
    for x_pos in X_pos:
        intersection_pos = x & x_pos
        if len(intersection_pos) < min_cardinality:  # the intersection is too small
            continue

        for x_neg in X_neg:  # count all negative examples that contain intersection_pos
            if (intersection_pos & x_neg) == intersection_pos:
                n_counters_pos += 1

    n_counters_neg = 0  # number of counter examples for negative intersections
    for x_neg in X_neg:
        intersection_neg = x & x_neg
        if len(intersection_neg) < min_cardinality:
            continue

        for x_pos in X_pos:  # count all positive examples that contain intersection_neg
            if (intersection_neg & x_pos) == intersection_neg:
                n_counters_neg += 1

    perc_counters_pos = n_counters_pos / len(X_pos)
    perc_counters_neg = n_counters_neg / len(X_neg)

    prediction = perc_counters_pos < perc_counters_neg
    return prediction

def predict_array(
        X: List[set], Y: List[bool],
        n_train: int, update_train: bool = True, use_tqdm: bool = False,
        predict_func: PREDICTION_FUNCTION_HINT = predict_with_generators
) -> Iterator[bool]:

    for i, x in tqdm(
        enumerate(X[n_train:]),
        initial=n_train, total=len(X),
        desc='Predicting step by step',
        disable=not use_tqdm,
    ):
        n_trains = n_train + i if update_train else n_train
        yield predict_func(x, X[:n_trains], Y[:n_trains]) 
    
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
