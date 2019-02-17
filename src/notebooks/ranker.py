import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

def get_top_ten(target):
    def objective(row):
        sum = 0
        for r, t in zip(row, target):
            sum += (r - t)**2
        return sum

    df = pd.read_csv('../data/processed/UTKFace_features_normalized.csv')
    
    df = df.apply(objective, axis=1)
    
    df = df.sort_values()
    
    smallest_list = df.nsmallest(10)
    
    smallest_list = smallest_list.keys().values.tolist()
    
    return smallest_list