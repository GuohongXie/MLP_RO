import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
#from mpl_toolkits.mplot3d.axes3d import Axes3D
import pandas as pd
from sklearn import preprocessing

#path = r"merged_csv\merged_csv.csv"
path = r"datasets\c19.csv"
df = pd.read_csv(path)
df = preprocessing.MinMaxScaler().fit_transform(df)
x = df[:,0].astype('float32')
y = df[:,1].astype('float32')
z = df[:,-2].astype('float32')


# view data
ax = plt.axes(projection='3d')
ax.scatter3D(x, y, z, c=z, cmap='Greens')
plt.show()