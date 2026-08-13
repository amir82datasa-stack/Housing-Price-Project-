import matplotlib.pyplot as plt
import pandas as pd
def box(df,column):
    plt.figure(figsize=(10,8))
    plt.boxplot(df[column],vert=False)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig("nmodar/bo.png")
    plt.close()




