
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 13:51:53 2021

@author: Filippos
"""
## Import Library
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

##    function to handle no numeric data.It converts them to int type 
def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df


#################################################################
###             DataBase 
#################################################################

## Data Input from cluster A. Database folder
df_all_list = []
folders = glob('C:/Users/Filippos/Desktop/Master/CSC8634-Cloud/data/cluster a')
folders = [s for s in folders if "csv" not in s]

for j in folders:
   ##print('j',j)
   csv = glob(j + '/*')
   for i in csv:
       print('load file:',i)
       dfi = pd.read_csv(i,sep='\t' )
       dat=dfi
       dat.columns = [ 'timestamp','packet','srcIP','desIP','srcPort','desPort','IP','srchost','deshost', 'srcrank','desrank','srpod','despod','intercluster','interdatacenter']

       df_all_list.append(dat)
     
df = pd.concat(df_all_list,axis=0)
df.columns = ['timestamp','packet','srcIP','desIP','srcPort','desPort','IP','srchost','deshost', 'srcrank','desrank','srpod','despod','intercluster','interdatacenter']
print ('import completed')
 
## Data Overview
df.dtypes
df.head()
#Convert date to normal date 
converted_df = pd.to_datetime(df['timestamp'], unit='s')
# Add new column to the DataFrame
dfDBase= df.assign(converted_time=converted_df)
print(dfDBase)
dfDBase.dtypes
dfDBase.isna().sum()
dfDBase.describe() 

## Data Analysis
dfDBase.corr()

## Convert Unix time timestamp  to normal time 
converted_df = pd.to_datetime(df['timestamp'], unit='s')
# Add new column to the DataFrame
dfDBase= df.assign(converted_time=converted_df)
## check for null values in the dataset 
dfDBase.isna().sum()
 
## Data Analysis
##dfDBase.corr()
##print('Correlation between intercluster   and interdatacenter is : {}'.format(round(dfDBase.corr()['intercluster']['interdatacenter'],3)))

#Correlation matrix with colorful indicators
plt.figure(figsize = (10,6))
sns.heatmap(df.corr(),annot=True,square=True,cmap='RdBu',vmax=1,vmin=-1)
plt.title('Correlations Between Variables',size=18);
plt.xticks(size=13)
plt.yticks(size=13)
plt.show()


#split the date in Day/Hour/Minute
dfDBase['Day']=dfDBase.converted_time.dt.day
dfDBase['Hour']=dfDBase.converted_time.dt.hour
dfDBase['Minute'] =dfDBase.converted_time.dt.minute

## plot showing the records per day 
plt.figure(figsize=(10,6))
sns.countplot(x = 'Day', data = dfDBase)
plt.title('Total Number of  requests per day in a  month',size=18)
plt.xlabel('Days',size=14)
plt.ylabel('No of requests',size=14)
plt.show()

## plot showing the records per Hour 
hours =[hour for hour, dfDBase in dfDBase[(dfDBase['Day'] == 1)].groupby('Hour')]                                      
plt.plot(hours,dfDBase[(dfDBase['Day'] == 1)].groupby(['Hour']).size())  
plt.xticks(hours)
plt.title('  Requests per Day In DB Server ',size=18)
plt.ylabel('No. of Requests',size=14)
plt.xlabel('Hours',size=14)
plt.grid()
plt.show()

## plot showing the records per Minute 
minutes  =[minute for minute , dfDBase in dfDBase[(dfDBase['Day'] == 1) & (dfDBase['Hour']==10)].groupby('Minute')]                                      
plt.plot(minutes,dfDBase[(dfDBase['Day'] == 1) & (dfDBase['Hour']==10)].groupby(['Minute']).size())  
#df2.groupby(['Hour']).size()
plt.xticks(minutes)
plt.grid()
plt.title('Requests per Hour in DB Server',size=18)
plt.ylabel('No of Requests',size=14)
plt.xlabel('Minutes',size=14)
plt.show()

## Plot showinng the Bandwith of traffic within a day
dfDBaseSample = dfDBase[(dfDBase['Day'] == 1)  ]
tempSample =dfDBaseSample.groupby(['Hour'], as_index=False, sort=False)['packet'].sum() #/1000000
tempSample['packet'] = tempSample['packet'] /1000000
sns.regplot(x = "Hour", y="packet", data=tempSample, ci=65,scatter=False, scatter_kws={"alpha": 0.2})
sns.lineplot(x = "Hour", y="packet", data=tempSample,)
plt.ylabel("Bandwith in Gbps ")
plt.xlabel('Hours')
plt.title('Total traffic in Gbps within a day in DB server')

## Pie chart-Total traffic in Gbps in interclusters in DD server
tempSample =dfDBaseSample.groupby(['intercluster'], as_index=False, sort=False)['packet'].sum() #/1000000
tempSample['packet'] = tempSample['packet'] /1000000
y = tempSample['packet'].values
labels = ['intra cluster', 'inter cluster' ]
#define Seaborn color palette to use
colors = sns.color_palette('pastel')[0:2]
#create pie chart
plt.pie(y, labels = labels  ,colors = colors, autopct='%.0f%%')
plt.title('Total traffic in Gbps in interclusters in DD server ')
plt.show()

import warnings
warnings.filterwarnings('ignore')

## Showing PDF, CDF for intercluster for the DAY 1 
sns.set_style("whitegrid")
interdatacenter_plot = sns.FacetGrid(dfDBaseSample, hue="intercluster", height=6)
interdatacenter_plot.map(sns.distplot, 'Hour').add_legend()
plt.xlabel("Hour")
plt.ylabel("PDF Density")
plt.title("Probability density function In intercluster In DB Server")
plt.show()

## Showing PDF, CDF for interdatacenter for the DAY 1 
sns.set_style("whitegrid")
interdatacenter_plot = sns.FacetGrid(dfDBaseSample, hue="interdatacenter", height=6)
interdatacenter_plot.map(sns.distplot, 'Hour').add_legend()
plt.xlabel("Hour")
plt.ylabel("PDF Density")
plt.title("Probability density function in interdatacenter In DB Server")
plt.show()

data_yes = dfDBaseSample.loc[dfDBaseSample["interdatacenter"] == 1];
data_no = dfDBaseSample.loc[dfDBaseSample["interdatacenter"] == 0];
hab =dfDBaseSample[['intercluster','interdatacenter','packet','Hour', 'Minute']]
plt.figure(figsize=(20,5))
for i, x in enumerate(list(hab.columns)[:-1]):
 
   plt.subplot(1, 4, i+1)
   counts, bin_edges = np.histogram(hab[x], bins=10, density=True)
   ##print(x,':{}'.format(bin_edges))
   pdf = counts/sum(counts)
   ##print(x,':{}'.format(pdf))
   cdf = np.cumsum(pdf)
   ##print(x,':{}'.format(cdf))
   ##print('===========================')
   plt.plot(bin_edges[1:], pdf, bin_edges[1:], cdf)
   plt.title("PDF vs CDF for Database Server Activity")
   plt.xlabel(x)
   plt.ylabel('PDF/CDF')
   
   dfDBaseSample = dfDBase[(dfDBase['Day'] == 1)  ]
a=dfDBaseSample['srcrank'].nunique()
b= dfDBaseSample['desrank'].nunique()
c=dfDBaseSample['srpod'].nunique()
d=dfDBaseSample['despod'].nunique()
e=dfDBaseSample['srcIP'].nunique()
z=dfDBaseSample['desIP'].nunique()
print ('Total usage of Requests/Racks/Pod for one day','\n Total source IP requested',e,'\n Total des IPs ' ,z)
print(' Total src rank used ', a   ,'\n Total des ranks used',b,'\n Total src Pod used',c,'\n Total des pod used',d)

##  convert the categorical values to int for the data of one day
dfDBaseSample=dfDBaseSample.drop(columns=['timestamp', 'IP'])
dfDBaseSample=dfDBaseSample.drop(columns=['converted_time' ])
dfDBaseSampleALLcol = handle_non_numerical_data(dfDBaseSample)
dfDBaseSampleALLcol.dtypes

hab =dfDBaseSampleALLcol[['srcrank','desrank','srpod','despod', 'Minute']]
plt.figure(figsize=(20,5))
for i, x in enumerate(list(hab.columns)[:-1]):
  
    plt.subplot(1, 6, i+1)
    counts, bin_edges = np.histogram(hab[x], bins=10, density=True)
    #print(x,':{}'.format(bin_edges))
    #pdf = counts/sum(counts)
    #print(x,':{}'.format(pdf))
    #cdf = np.cumsum(pdf)
    #print(x,':{}'.format(cdf))
    #print('===========================')
    plt.plot(bin_edges[1:], pdf, bin_edges[1:], cdf)
    plt.title("Database Server Activity")
    plt.xlabel(x)
    plt.ylabel('PDF/CDF')
    
# Hour group  
    Hour_group = ['Morning','Afternoon','Night']
## Declare a function for splitting the day into morning,night,afternoon

def label_Hour(row):
    if row['Hour'] < 12:
        return Hour_group[0]
    elif row['Hour'] < 18:
        return Hour_group[1]
    elif row['Hour'] >= 18:
        return Hour_group[2]

grouped_single = dfDBaseSampleALLcol.groupby('srpod').size()     #  .agg({'packet': ['mean', 'min', 'max']})
print('Source Pod Request:',grouped_single)

## we see that from scrpod  3 uses many despod requests 
tempSample =dfDBaseSampleALLcol.groupby(['srpod','Hour'], as_index=False, sort=False)['despod'].count()
tempSample['Hour_group'] = tempSample.apply(lambda row: label_Hour(row), axis=1)
sns.FacetGrid(tempSample, col = "Hour_group").map(plt.scatter, "srpod", "despod", alpha =0.2).add_legend()
plt.xlabel('source Pod')

##  plot srcpod vs despod vs Hour 
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
x = dfDBaseSampleALLcol['srpod']
y = dfDBaseSampleALLcol['despod']
z = dfDBaseSampleALLcol['Hour']
ax.scatter(x, y, z)
ax.set_xlabel("srpod")
ax.set_ylabel("despod")
ax.set_zlabel("Hour")
plt.show()

dfDBaseSample = dfDBase[(dfDBase['Day'] == 1)] 
tempSample = pd.DataFrame({'count' : dfDBaseSample.groupby( ['srchost','deshost'], as_index=False, sort=False ).size()}).reset_index()
## tempSample = tempSample.sort_values(by=['size'] , ascending=False)
a= tempSample['srchost'].nunique()
b= tempSample['deshost'].nunique()
print(' Total number of source Host ', a   ,'\n Total number of destination Host used',b )
tempgraph= tempSample.head(10)
c=tempgraph['count'].sum()
d=tempSample['count'].sum()
print(' The top 10 source host have total requests: ',c   ,'\n Total Requests from source to Dest. Host are :',d )
tempgraph.head(9)

##sns.regplot(x = "srchost", y="deshost", data=tempSample, ci=65,scatter=False, scatter_kws={"alpha": 0.2})
sns.lineplot(x = "srchost", y="deshost", data=tempgraph,)
plt.ylabel("Dest. Host")
plt.xlabel('source Host')
plt.title('Total traffic between these sources in DB Server')

##  MODEL AND EVALUATION For Database server
# visualize the data
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from time import time
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


## data  are split to train and test set.
## 6 variables were used in train set.
dfML = dfDBaseSampleALLcol[["packet","Hour","srpod","despod","srcrank","desrank","interdatacenter"]]
# Split-out validation dataset
array = dfML.values
X = array[:,0:6]
y = array[:,6]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LR2',LogisticRegression(penalty='l2',solver = 'lbfgs', C = 1)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('RFC', RandomForestClassifier(criterion = 'entropy', random_state = 42)))
models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn

import warnings
warnings.filterwarnings('ignore')
print (' 7 Models are  processing ...')
results = []
names = []
for name, model in models:
     
    t0 = time()
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    f = '{0:.2f}'.format(time()-t0) 
    print('Training: %s: %f (%f) takes %s seconds' % (name, cv_results.mean(), cv_results.std(),  f))
  
# Compare Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Model algorithm Comparison')
pyplot.ylabel('Accuracy')
pyplot.show()

## Building a Random Forest Model 
rfc = RandomForestClassifier(criterion = 'entropy', random_state = 42)
print ('processing...')
rfc.fit(X_train, Y_train)
print ('model run')

predictionsrfc =rfc.predict(X_validation)
print ('Predictions on test set is completed')

## Evaluation of the method 
print('Accuracy: ' , accuracy_score(Y_validation, predictionsrfc))
print('confusion_matrix \n', confusion_matrix(Y_validation, predictionsrfc))
print('classification_report\n',classification_report(Y_validation, predictionsrfc))  

# Creating a Confusion Matrix
cm = confusion_matrix(Y_validation, predictionsrfc)
df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (28,20))
fig, ax = plt.subplots()
sns.set(font_scale=1.4)
sns.heatmap(df_cm, annot=True, fmt='g')
class_names=[0,1]
tick_marks = np.arange(len(class_names))
plt.tight_layout()
plt.title('Confusion Matrix-Database server \n', y=1.1)
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
ax.xaxis.set_label_position("top")
plt.ylabel('Actual label\n')
plt.xlabel('Predicted label\n')

# summarize the first 5 cases
for i in range(5):
    print('%s => %d (expected %d)' % (X_validation[i].tolist(), predictionsrfc[i], Y_validation[i]))
    

## Building a Decision tree Model 
dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 42)
dt.fit(X_train, Y_train)

feature_importance=pd.DataFrame({
    'dt':dt.feature_importances_,
    'rfc':rfc.feature_importances_
},index=dfML.drop(columns=['interdatacenter']).columns)
feature_importance.sort_values(by='dt',ascending=True,inplace=True)

index = np.arange(len(feature_importance))
fig, ax = plt.subplots(figsize=(18,8))
dt_feature=ax.barh(index+0.4,feature_importance['dt'],0.4,color='lightgreen',label='Decision Tree')
rfc_feature=ax.barh(index,feature_importance['rfc'],0.4,color='purple',label='Random Forest')
ax.set(yticks=index+0.4,yticklabels=feature_importance.index)

ax.legend()
plt.show()

#################################################################
###             HADOOP 
#################################################################

# Loading dat from Cluster C
df_all_list = []
folders = glob('C:/Users/Filippos/Desktop/Master/CSC8634-Cloud/data/CLUSTER C')

folders = [s for s in folders if "csv" not in s]
for j in folders:
   ##print('j',j)
   csv = glob(j + '/*')
   for i in csv:
       print('files loaded',i)
       dfi = pd.read_csv(i,sep='\t' )
       dat=dfi
       dat.columns = [ 'timestamp','packet','srcIP','desIP','srcPort','desPort','IP','srchost','deshost', 'srcrank','desrank','srpod','despod','intercluster','interdatacenter']
       df_all_list.append(dat)
     
df_all_hadoop = pd.concat(df_all_list,axis=0)
df_all_hadoop.columns = ['timestamp','packet','srcIP','desIP','srcPort','desPort','IP','srchost','deshost', 'srcrank','desrank','srpod','despod','intercluster','interdatacenter']
print('Data imported')

## convert the timestamp(unix time) to normal time and display the data 
converted_df = pd.to_datetime(df_all_hadoop['timestamp'], unit='s')
# Add new column to the DataFrame
dfHadoop = df_all_hadoop.assign(converted_time=converted_df)
##print(dfHadoop)
##dfHadoop.dtypes
dfHadoop.head()
 
dfHadoop.size

#Correlation matrix with colorful indicators
plt.figure(figsize = (10,6))
sns.heatmap(dfHadoop.corr(),annot=True,square=True,cmap='RdBu',vmax=1,vmin=-1)
plt.title('Correlations Between Variables in Hadoop Server',size=18);
plt.xticks(size=13)
plt.yticks(size=13)
plt.show()

# Split the dataset in days/Hour/Minutes
dfHadoop['Day']=dfHadoop.converted_time.dt.day
dfHadoop['Hour']=dfHadoop.converted_time.dt.hour
dfHadoop['Minute'] =dfHadoop.converted_time.dt.minute

dfHadoop.groupby(['Day'],sort = True).size()

days =[day for day, dfHadoop in dfHadoop.groupby('Day')]
daytrend = dfHadoop.groupby(['Day'],sort = True).size()
fig = plt.figure(figsize = (10, 5))
# creating the bar plot
plt.bar(days, daytrend, color ='maroon', width = 0.2)
plt.xlabel("Days of a month")
plt.ylabel("No. of Requests ")
plt.title("Hadoop  inquiries per day ")
plt.show()

hours =[hour for hour, df in dfHadoop[(dfHadoop['Day'] == 1)].groupby('Hour')]                                  
plt.plot(hours,dfHadoop[(dfHadoop['Day'] == 1)].groupby(['Hour'],sort = True).size())  
#df2.groupby(['Hour']).size()
plt.xticks(hours)
plt.xlabel("Hours in a Day")
plt.ylabel("No. of Requests ")
plt.title('Traffic in one day -Hadoop Server')
plt.grid()
plt.show()

df2sampleHadoop =  dfHadoop[(dfHadoop['Day'] == 1)]
a=df2sampleHadoop['srcrank'].nunique()
b= df2sampleHadoop['desrank'].nunique()
c=df2sampleHadoop['srpod'].nunique()
d=df2sampleHadoop['despod'].nunique()
e=df2sampleHadoop['srcIP'].nunique()
z=df2sampleHadoop['desIP'].nunique()
print ('Total usage of Requests/Racks/Pod for one day','\n Total source IP requested',e,'\n Total des IPs ' ,z)
print(' Total src rank used ', a   ,'\n Total des ranks used',b,'\n Total src Pod used',c,'\n Total des pod used',d)

tempSample =df2sampleHadoop.groupby(['Hour'], as_index=False, sort=False)['packet'].sum() #/1000000
tempSample['packet'] = tempSample['packet'] /1000000
sns.regplot(x = "Hour", y="packet", data=tempSample, ci=65,scatter=False, scatter_kws={"alpha": 0.2})
sns.lineplot(x = "Hour", y="packet", data=tempSample,)
plt.ylabel("Bandwith on Gbps")
plt.xlabel('Hours')
plt.title('Total traffic in Gbps in one day -Hadoop Server')


tempSample =df2sampleHadoop.groupby(['intercluster'], as_index=False, sort=False)['packet'].sum() #/1000000
tempSample['packet'] = tempSample['packet'] /1000000
y = tempSample['packet'].values
labels = ['inter cluster', 'intra cluster' ]
#define Seaborn color palette to use
colors = sns.color_palette('pastel')[0:2]
#create pie chart
plt.title('Traffic  in Gbps  in  cluster -Hadoop Server')
plt.pie(y, labels = labels  ,colors = colors, autopct='%.0f%%')
plt.show()


tempSample =df2sampleHadoop.groupby(['interdatacenter'], as_index=False, sort=False)['packet'].sum() #/1000000
tempSample['packet'] = tempSample['packet'] /1000000
y = tempSample['packet'].values
labels = ['inter datacenter', 'intra datacenter' ]
#define Seaborn color palette to use
colors = sns.color_palette('pastel')[0:2]
#create pie chart
plt.title('Hadoop- Total traffic in datacenter per day')
plt.pie(y, labels = labels  ,colors = colors, autopct='%.0f%%')
plt.show()


import warnings
warnings.filterwarnings('ignore')
## Showing pdf ,CDF for intercluster for the DAY 1 
sns.set_style("whitegrid")
interdatacenter_plot = sns.FacetGrid(df2sampleHadoop, hue="intercluster", height=6)
interdatacenter_plot.map(sns.distplot, 'Hour').add_legend()
plt.xlabel("Hour")
plt.ylabel("PDF Density")
plt.title("Probability density function In intercluster In Hadoop Server")
plt.show()


sns.set_style("whitegrid")
interdatacenter_plot = sns.FacetGrid(df2sampleHadoop, hue="interdatacenter", height=6)
interdatacenter_plot.map(sns.distplot, 'Hour').add_legend()
plt.xlabel("Hour")
plt.ylabel("PDF Density")
plt.title("Probability density function in interdatacenter In Hadoop Server")
plt.show()


hab =df2sampleHadoop[['intercluster','interdatacenter','packet','Hour', 'Minute']]
plt.figure(figsize=(20,5))
for i, x in enumerate(list(hab.columns)[:-1]):
 
   plt.subplot(1, 4, i+1)
   counts, bin_edges = np.histogram(hab[x], bins=10, density=True)
   ##print(x,':{}'.format(bin_edges))
   pdf = counts/sum(counts)
   ##print(x,':{}'.format(pdf))
   cdf = np.cumsum(pdf)
   ##print(x,':{}'.format(cdf))
   ##print('===========================')
   plt.plot(bin_edges[1:], pdf, bin_edges[1:], cdf)
   plt.title("PDF vs CDF for Hadoop Server Activity")
   plt.xlabel(x)
   plt.ylabel('PDF/CDF')

## Convert all the non-numeric variables to numeric using the function 'handle_non_numerical_data'
df2sampleHadoop=df2sampleHadoop.drop(columns=['timestamp', 'IP'])
df2sampleHadoop=df2sampleHadoop.drop(columns=['converted_time' ])
df2sampleHadoopALLcol = handle_non_numerical_data(df2sampleHadoop)
#pd.get_dummies(dfminute)
df2sampleHadoopALLcol.dtypes

hab =df2sampleHadoopALLcol[['srcrank','desrank','srpod','despod', 'Minute']]
labels =['source rack','des rack','soure pod','des pod' ,'Minute']
plt.figure(figsize=(20,5))
for i, x in enumerate(list(hab.columns)[:-1]):
 
    plt.subplot(1, 6, i+1)
    counts, bin_edges = np.histogram(hab[x], bins=10, density=True)
   # print(x,':{}'.format(bin_edges))
    pdf = counts/sum(counts)
    #print(x,':{}'.format(pdf))
    cdf = np.cumsum(pdf)
    #print(x,':{}'.format(cdf))
   # print('===========================')
    plt.plot(bin_edges[1:], pdf, bin_edges[1:], cdf)
    plt.xlabel(labels[i])
    plt.ylabel('PDF / CDF')
    plt.title('PDF/CDF-Hadoop Server')



#   MODEL IN HADOOP using time series to predict the traffic within the day.
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARMA

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
sns.set()

##  create a dataset regarding Hours and traffic 
dfmodelHadoop = pd.DataFrame()
hours =[hour for hour, df in dfHadoop[(dfHadoop['Day'] == 1)].groupby('Hour')]
dfmodelHadoop['Hits'] = dfHadoop[(dfHadoop['Day'] == 1)].groupby('Hour',sort=True).size()
dfmodelHadoop['hours'] = hours
dfmodelHadoop.head()

dfmodelHadoop.isna().sum()
dfmodelHadoop=dfmodelHadoop.dropna()

## split the data for one day.Train set all the traffic before 18:00 and test set all the traffic after 18:00.
train = dfmodelHadoop[dfmodelHadoop.hours <= 18]  
train = train.set_index('hours')                                                    
test  = dfmodelHadoop[dfmodelHadoop.hours > 18]
test = test.set_index('hours')  

plt.plot(train, color = 'black', label = 'Training')
plt.plot(test, color = 'red', label = 'Testing')
plt.ylabel('Hits in DB')
plt.xlabel('Hours')
plt.xticks(rotation=45)
plt.title("Train/Test split for Hadoop Data")
plt.legend()
plt.show()


#  Build Models based on time series

y = train['Hits']
ARMAmodel = SARIMAX(y, order = (5, 1, 1))
ARMAmodel = ARMAmodel.fit()
print ('Model process Completed in ')

y_pred = ARMAmodel.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha = 0.05)
y_pred_df["Predictions"] = ARMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = test.index
y_pred_outARMA = y_pred_df["Predictions"]
plt.plot(y_pred_outARMA, color='green', label = 'ARMA  Predictions')
plt.legend()

import numpy as np
from sklearn.metrics import mean_squared_error

arma_rmse = np.sqrt(mean_squared_error(test["Hits"].values, y_pred_df["Predictions"]))
print("ARMA RMSE: ",arma_rmse)

from statsmodels.tsa.arima.model import ARIMA
ARIMAmodel = ARIMA(y, order = (2,1, 1))
ARIMAmodel = ARIMAmodel.fit()
print ('ARIMA model completed')

y_pred = ARIMAmodel.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha = 0.05)
y_pred_df["Predictions"] = ARIMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = test.index
y_pred_outARIMA = y_pred_df["Predictions"]
plt.plot(y_pred_outARIMA, color='Yellow', label = 'ARIMA Predictions')
plt.legend()

from sklearn.metrics import mean_squared_error
arma_rmse = np.sqrt(mean_squared_error(test["Hits"].values, y_pred_df["Predictions"]))
print("ARIMA RMSE: ",arma_rmse)

##   SARIMA
SARIMAXmodel = SARIMAX(y, order = (3, 2, 2), seasonal_order=(2,2,2,15))
SARIMAXmodel = SARIMAXmodel.fit()
print('SARIMA: model completed')

y_pred = SARIMAXmodel.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha = 0.05)
y_pred_df["Predictions"] = SARIMAXmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = test.index
y_pred_outSARIMA = y_pred_df["Predictions"]
plt.plot(y_pred_outSARIMA, color='Blue', label = 'SARIMA Predictions')
plt.legend()
plt.show()

import numpy as np
from sklearn.metrics import mean_squared_error
arma_rmse = np.sqrt(mean_squared_error(test["Hits"].values, y_pred_df["Predictions"]))
print("SARIMA RMSE: ",arma_rmse)


plt.plot(train, color = 'black', label = 'Training Set')
plt.plot(test, color = 'red', label = 'Testing Set')
plt.ylabel('No of Requests')
plt.xlabel('Hours')
plt.xticks(rotation=45)
plt.title("Train/Test/Predictions  for Hadoop Server")
plt.plot(y_pred_outARMA, color='green', label = 'ARMA  Predictions')
plt.plot(y_pred_outARIMA, color='Yellow', label = 'ARIMA Predictions')
plt.plot(y_pred_outSARIMA, color='Blue', label = 'SARIMA Predictions')
plt.legend()
plt.show()


