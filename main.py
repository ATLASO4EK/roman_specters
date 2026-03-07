import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

script_path = os.getcwd()
data_path = os.path.join(script_path, 'data')
path = os.path.join(data_path, 'control', 'mk1')

files = os.listdir(path)
file = os.path.join(path, files[0])

df = pd.read_csv(file, delim_whitespace=True)

# df.iloc[:,2:4]

x, y = df['#Wave'].tolist(), df['#Intensity'].tolist()

plt.plot(x, y)
plt.show()