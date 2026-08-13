import pandas as pd
from pandas import DataFrame


def transform_outlir(df:DataFrame):
    dff = df.copy()
    keep_mask = pd.Series(True, index=dff.index)
    candidate_outlier_cols = ["SalePrice", "LotArea", "GrLivArea", "TotalBsmtSF"]
    for i in candidate_outlier_cols:
        if i  not in dff.columns:
            continue
        q1 = dff[i].quantile(0.25)
        q3 = dff[i].quantile(0.75)
        iqr = q3 - q1
        lower = q1- 1.5 * iqr
        higher = q3 + 1.5 * iqr
        mask = (dff[i]  >= lower) & (dff[i] <= higher)
        keep_mask = keep_mask & mask
    return dff[keep_mask]


