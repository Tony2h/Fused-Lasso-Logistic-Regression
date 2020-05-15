import pandas as pd
from sklearn.model_selection import train_test_split

def get_data(standard=1, normal=1):
    # standard=1:Standardize the data
    # normal=1:normalize the data
    data = pd.read_csv('pd_speech_features_noID.csv', header=None)
    y = data.values[:, -1]
    if standard:
        data = (data - data.mean()) / data.std()
    if normal:
        data = (data - data.min(axis=0)) / (data.max(axis=0)- data.min(axis=0))
    X = data.values[:, :-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, y_train, X_test, y_test
