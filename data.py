import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

url = "https://archive.ics.uci/edu/ml/machine-learning-databases/iris/iris.data"

# Assign column names to the dataset
names = ['sepal-lenght', 'sepal-width', 'petal-lenght', 'petal-width', 'Class']

# Read dataset
df = pd.read_csv(url, names=names)