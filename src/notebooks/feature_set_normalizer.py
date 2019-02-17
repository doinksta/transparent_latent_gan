import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

df = pd.read_csv('../data/processed/UTKFace_features.csv')

# df = pd.DataFrame({'':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'a':[1,3,5,7,4,5,6,4,7,8,9], 'b':[3,5,6,2,4,6,7,8,7,8,9]})

df = df.drop(df.columns[0], axis=1)


means = df.mean(axis=0)
stds = df.std(axis=0)


data = df.values
data = np.transpose(data)

normed = []

for i, col in zip(range(len(data)), data):
    new_col = []
    
    for val in col:
        new_col.append((val - means[i]) / stds[i])
        
    normed.append(new_col)

normed = np.transpose(normed)

new_df = pd.DataFrame(normed)

new_df.to_csv("../data/processed/UTKFace_features_normalized.csv")


#df['0'].sort
#df.sort_values('0')
#Now grab the indices in order