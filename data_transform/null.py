import numpy as np
from pandas import DataFrame
from sklearn.impute import SimpleImputer

def transform_null(df:DataFrame)    :
    cols = df.isnull().mean().sort_values(ascending=False)
    cols = cols[cols > 0.4].index.tolist()
    df.drop(columns=cols, inplace=True)
    print("جذف ستون  های خالی",df.shape)


def transform_imputar(df:DataFrame):
    numbar_cols = df.select_dtypes(include=np.number).columns.tolist()
    cacgore_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    df[numbar_cols] = SimpleImputer(strategy="median").fit_transform(df[numbar_cols])
    df[cacgore_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cacgore_cols])
    return df,numbar_cols, cacgore_cols




