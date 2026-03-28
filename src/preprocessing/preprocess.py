
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def preprocess(df: pd.DataFrame):
    """Preprocess dataset (encode + normalize)."""
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X.fillna(0, inplace=True)

    if y.dtype == 'object':
        y = pd.Series(LabelEncoder().fit_transform(y))

    X = pd.DataFrame(MinMaxScaler().fit_transform(X))
    return X, y
