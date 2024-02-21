import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
dataset = pd.read_csv("advertising.csv")
dataset.head()
dataset.shape
dataset.isna().sum() '''check for missing values'''
dataset.duplicated().any() '''check for duplicate values'''
fig,axs = plt.subplots(2,figsize=(5,5))
