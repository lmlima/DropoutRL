import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from scripts_ufes.data.evasao.gen_discrete_student import plot3d_state_discrete


def fig3():
    df = pd.read_pickle("tmp/data/xmeans/2/pca_data_discretization.dataframe.pkl")
    view = {'elev': 62, 'azim': -32}

    plot3d_state_discrete(df, output='/tmp', view=view)
