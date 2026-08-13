from shlex import join
from xml.etree.ElementInclude import include

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fontTools.designspaceLib import split
from narwhals.testing.asserts import frame
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from data_transform.normal import transform_norm, transform_dublin
from data_transform.null import transform_null, transform_imputar
from data_transform.outlir import transform_outlir
from nmudar.boxplat import box
from nmudar.histogram import histo
from nmudar.hitmap import histogra


df = pd.read_csv("data/train.csv")
print("تعداد ستون ها و ردیف ها ",df.shape)

df_null = df.copy()

transform_null(df_null)
df_null, numbar_cols, cacgore_cols = transform_imputar(df_null)
print("جای گذاری انجام شد ",df_null.shape)

# ^^^^^^^^^^^^^^^  عملیات داده های خالی  ^^^^^^^^^^^^^^^^
df_null = transform_outlir(df_null)
print("حذف ستون های خالی ",df_null.shape)
# ^^^^^^^^^^^^^^^  عملیات داده پرت   ^^^^^^^^^^^^^^^^
for col in cacgore_cols:
         df_null[col] = df_null[col].map(transform_norm)


dubli = transform_dublin(df_null)
df_null.drop(columns=dubli, inplace=True)
print("حذف ستون تکراری  ",df_null.shape)
# ^^^^^^^^^^^^^^^  عملیات داده تکراری   ^^^^^^^^^^^^^^^^


dff = df_null.copy()
numbar_histo = dff.select_dtypes(include=[np.number])
coor = numbar_histo.corr()
histogra(coor)
print(dff.shape)


if "SalePrice" in dff.columns:
    histo(dff,"SalePrice")
    box(dff,"SalePrice")

        # ^^^^^^^^^^^^^^^  عملیات نمودار ها    ^^^^^^^^^^^^^^^^

    if "YearBuilt" in dff.columns and "YrSold" in dff.columns:
        dff["house_age"] = dff["YrSold"] - dff["YearBuilt"]

    if all([c in dff.columns for c in ["SalePrice", "YearBuilt"]]):
        dff["remod_age"] = dff["SalePrice"] - dff["YearBuilt"]

    wanted = ["GrLivArea", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]
    comp_cols = [c for c in wanted if c in dff.columns]
    if comp_cols:
        dff["total_sf"] = dff[comp_cols].sum(axis=1)

#
numrec_pca = dff.select_dtypes(include=np.number).columns.tolist()
features_pca = [c for c in numrec_pca if c not  in ["Id","SalePrice"]]

df_pca = dff[features_pca].copy()
df_pca = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(df_pca),columns=features_pca)
Standard_pca = StandardScaler()
X_scaled = Standard_pca.fit_transform(df_pca)

pca = PCA(n_components=0.95, svd_solver="full",random_state=42)
x_pca = pca.fit_transform(X_scaled)
print(f"تعداد ستون‌های جدیدی که ساخته شده را چاپ می‌کند {x_pca.shape[1]} ")
e_v_r = pca.explained_variance_ratio_
# واریانس_نسبت_تبیین_شده
cum_var = np.cumsum(e_v_r)
# np.cumsum: مخفف Cumulative Sum است. این تابع مقادیر بالا را با هم جمع می‌کند تا “
# واریانس تجمعی” را به دست بیاورد.
print( f" آخرین مقدار در آرایه تجمعی را می‌گیرد ( {(cum_var[-1]*100):.1f}%")



df_stand = dff.copy()
num_cols_sc = [c for c in df_stand.select_dtypes(include=[np.number]).columns
               if c not in ["SalePrice"]]
standard_1 = StandardScaler()
df_x = df_stand.copy()
df_x[num_cols_sc] = standard_1.fit_transform(df_x[num_cols_sc])
print(df_x.shape)



