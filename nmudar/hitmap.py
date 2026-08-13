from turtledemo.clock import tick

import matplotlib.pyplot as plt
import pandas as pd

def histogra(coor):
    plt.figure(figsize=(10,8))
    plt.imshow(coor,aspect='auto')
    plt.colorbar()
    plt.xticks(ticks=range(len(coor.columns)),labels=coor.columns,rotation=90,fontsize=10)
    plt.yticks(ticks=range(len(coor.columns)),labels=coor.columns,fontsize=10)
    plt.tight_layout()
    plt.savefig("nmudar\hitmap.png")
    plt.close()