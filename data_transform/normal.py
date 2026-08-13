import pandas as pd
from pandas import DataFrame

def transform_norm(col):
    if pd.isnull(col) or not isinstance(col,str):
        return col
    return "".join(col.strip().capitalize().split())


def transform_dublin(df:pd.DataFrame):
    dublict = []
    x = len(df)
    for i in df.columns:
        u = df[i].nunique()
        if (u / max(x,1)) < 0.0035:
            dublict.append(i)
    return dublict
