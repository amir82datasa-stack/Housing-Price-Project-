import matplotlib.pyplot as plt
import pandas as pd



def histo(df,column):
    plt.figure(figsize=(10,8))
    df[column].hist(bins=50)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig("nmodar/histogram.png")
    plt.close()




