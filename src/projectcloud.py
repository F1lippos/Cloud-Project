# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 13:51:53 2021

@author: georg
"""

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # linear algebra
import os 
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob
#import tensorflow as tf
from sklearn.model_selection import train_test_split
from matplotlib.gridspec import GridSpec

dat = pd.read_csv('C:/Users/Filippos/Desktop/Master/CSC8634-Cloud/14615503_1224523740902654_6852172664916672512_n',sep='\t' )
dat.head()
dat.describe()

df =dat

df.columns = ['timestamp','packet','srcIP','desIP','srcPort','desPort','IP','srchost','deshost', 'srcrank','desrank','srpod','despod','intercluster','interdatacenter']
 
 
converted_df = pd.to_datetime(df['timestamp'], unit='s')

# Add new column to the DataFrame
 
df2 = df.assign(converted_time=converted_df)
print(df2)


# to print the total records 
index = df2.index
number_of_rows = len(index) #find length of index.
print(number_of_rows)

# counting unique values of timestamp that the datamart has request 
n = len(pd.unique(df2['timestamp']))
print("No.of.unique values :", n)
print(24*60*60)



print(df2.groupby('timestamp').count())
xcount =df2.groupby('timestamp').size()
# decsribe the hints on the datacenter of request in facebook
xcount.describe()


analysis = df2.groupby('timestamp').count()