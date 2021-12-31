# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 13:51:53 2021

@author: georg
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

##    function 
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
###             WEB 
#################################################################

## Data Input
df_all_list = []
##folders = glob('C:/Users/Filippos/Desktop/Master/CSC8634-Cloud/data/cluster a')
folders = glob('C:/Users/Filippos/Desktop/Master/CSC8634-Cloud/data/cluster a')
folders = [s for s in folders if "csv" not in s]

for j in folders:
   print('j',j)
   csv = glob(j + '/*')
   for i in csv:
       print('i',i)
       dfi = pd.read_csv(i,sep='\t' )
       dat=dfi
       dat.columns = [ 'timestamp','packet','srcIP','desIP','srcPort','desPort','IP','srchost','deshost', 'srcrank','desrank','srpod','despod','intercluster','interdatacenter']

       df_all_list.append(dat)
     
df = pd.concat(df_all_list,axis=0)
df.columns = ['timestamp','packet','srcIP','desIP','srcPort','desPort','IP','srchost','deshost', 'srcrank','desrank','srpod','despod','intercluster','interdatacenter']

 
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

print('Correlation between age and charges is : {}'.format(round(dfDBase.corr()['intercluster']['interdatacenter'],3)))
##print('Correlation between age and charges is : {}'.format(round(dfDBase.corr()['srcrank']['desrank'],3)))

#Correlation matrix with colorful indicators
plt.figure(figsize = (10,6))
sns.heatmap(df.corr(),annot=True,square=True,cmap='RdBu',vmax=1,vmin=-1)
plt.title('Correlations Between Variables',size=18);
plt.xticks(size=13)
plt.yticks(size=13)
plt.show()



#plot per momnth the records.
dfDBase['Day']=dfDBase.converted_time.dt.day
dfDBase['Hour']=dfDBase.converted_time.dt.hour
dfDBase['Minute'] =dfDBase.converted_time.dt.minute



#days =[day for day, df in dfDBase.groupby('Day')]
#daytrend = dfDBase.groupby(['Day']).size()
#fig = plt.figure(figsize = (10, 5))
# creating the bar plot
#plt.bar(days, daytrend, color ='maroon', width = 0.2)
#plt.xlabel("Days of a month")
#plt.ylabel("No. of hits ")
#plt.title("Wen inquiries per day ")
#plt.show()


## plot showing the records per day 
plt.figure(figsize=(10,6))
sns.countplot(x = 'Day', data = dfDBase)
plt.title('Total Number of Male and Female',size=18)
plt.xlabel('Days',size=14)
plt.show()


# I get the maximum daye of records 
dfDBaseSample = dfDBase[(dfDBase['Day'] == 1)]
dfDBaseSample.groupby(['Day']).size()

## plot showing the records per Hour 
hours =[hour for hour, dfDBase in dfDBase[(dfDBase['Day'] == 1)].groupby('Hour')]                                      
plt.plot(hours,dfDBase[(dfDBase['Day'] == 1)].groupby(['Hour']).size())  
#df2.groupby(['Hour']).size()
plt.xticks(hours)
plt.grid()
plt.show()


## plot showing the records per Minute 
minutes  =[minute for minute , dfDBase in dfDBase[(dfDBase['Day'] == 1) & (dfDBase['Hour']==10)].groupby('Minute')]
#hours =[hour for hour, df in dfDBase.groupby('Hour')]
#plt.plot(hours,df2[(df2['Day'] == 1)].groupby(['Hour']).size())                                         
plt.plot(minutes,dfDBase[(dfDBase['Day'] == 1) & (dfDBase['Hour']==10)].groupby(['Minute']).size())  
#df2.groupby(['Hour']).size()
plt.xticks(minutes)
plt.grid()
plt.show()


dfDBase[(dfDBase['Day'] == 1) & (dfDBase['Hour']==10)].groupby(['Minute']).size()
dfHour = dfDBase[(dfDBase['Day'] == 1) & (dfDBase['Hour']==10)]
dfminute = dfDBase[(dfDBase['Day'] == 1) & (dfDBase['Hour']==10) & (dfDBase['Minute']== 8)]
 

dfminute.dtypes
dfminute.groupby(['srcIP']).count()
dfminute.groupby(['srcIP']).size()

dfminute.groupby(['packet','intercluster','interdatacenter']).size()
dfminute.groupby(['packet']).sum()



## Showing pdf ,CDF for intercluster for the DAY 1 
#pdf for "intercluster" input variable
sns.set_style("whitegrid")
interdatacenter_plot = sns.FacetGrid(dfDBaseSample, hue="intercluster", height=6)
interdatacenter_plot.map(sns.distplot, 'Hour').add_legend()
plt.xlabel("PDF of intercluster")
plt.ylabel("Counts")
plt.title("Probability density function between intercluster vs Counts")
plt.show()


#splitting the data frame into two data frames having status as 1,2
data_yes = dfDBaseSample.loc[dfDBaseSample["intercluster"] == 1];
data_no = dfDBaseSample.loc[dfDBaseSample["intercluster"] == 0];
#data_yes = all the data points in status where values are 1
#data_no = all the data points in status where values are 0
#status = 1
counts, bin_edges = np.histogram(data_yes['Hour'], bins=12,density = True)
#pdf gives you the total percent of output values (1 or 2) present #for the selected x value (which is age in our case)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], cdf,label=1)
# status = 0
counts, bin_edges = np.histogram(data_no['Hour'], bins=12,density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], cdf,label=2)
plt.title("CDF for intercluster of web")
plt.xlabel("Intercluster of web")
plt.ylabel('cumulative % of intercluster')
plt.legend()

## Showing pdf ,CDF for interdatacenter for the DAY 1 
#pdf for "intercluster" input variable
sns.set_style("whitegrid")
interdatacenter_plot = sns.FacetGrid(dfDBaseSample, hue="interdatacenter", height=6)
interdatacenter_plot.map(sns.distplot, 'Hour').add_legend()
plt.xlabel("PDF of interdatacenter")
plt.ylabel("Counts")
plt.title("Probability density function between interdatacenter vs Counts")
plt.show()


#splitting the data frame into two data frames having status as 1,2
data_yes = dfDBaseSample.loc[dfDBaseSample["interdatacenter"] == 1];
data_no = dfDBaseSample.loc[dfDBaseSample["interdatacenter"] == 0];
#data_yes = all the data points in status where values are 1
#data_no = all the data points in status where values are 0
#status = 1
counts, bin_edges = np.histogram(data_yes['Hour'], bins=12,density = True)
#pdf gives you the total percent of output values (1 or 2) present #for the selected x value (which is age in our case)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], cdf,label=1)
# status = 0
counts, bin_edges = np.histogram(data_no['Hour'], bins=12,density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], cdf,label=2)
plt.title("CDF for interdatacenter of web")
plt.xlabel("interdatacenter of web")
plt.ylabel('cumulative % of interdatacenter')
plt.legend()


## plot intercluster','interdatacenter'
hab =dfDBaseSample[['intercluster','interdatacenter','packet', 'Minute']]
plt.figure(figsize=(20,5))
for i, x in enumerate(list(hab.columns)[:-1]):
 
   plt.subplot(1, 3, i+1)
   counts, bin_edges = np.histogram(hab[x], bins=10, density=True)
   print(x,':{}'.format(bin_edges))
   pdf = counts/sum(counts)
   print(x,':{}'.format(pdf))
   cdf = np.cumsum(pdf)
   print(x,':{}'.format(cdf))
   print('===========================')
   plt.plot(bin_edges[1:], pdf, bin_edges[1:], cdf)
   plt.xlabel(x)



hab =dfDBaseSample[['intercluster','interdatacenter','Hour', 'Minute']]
plt.figure(figsize=(20,5))
for i, x in enumerate(list(hab.columns)[:-1]):
 
   plt.subplot(1, 3, i+1)
   counts, bin_edges = np.histogram(hab[x], bins=10, density=True)
   print(x,':{}'.format(bin_edges))
   pdf = counts/sum(counts)
   print(x,':{}'.format(pdf))
   cdf = np.cumsum(pdf)
   print(x,':{}'.format(cdf))
   print('===========================')
   plt.plot(bin_edges[1:], pdf, bin_edges[1:], cdf)
   plt.xlabel(x)





grouped_single = dfminute.groupby('srcIP').agg({'packet': ['mean', 'min', 'max']})
print(grouped_single)

grouped_single = dfminute.groupby('srpod').agg({'despod': ['mean', 'min', 'max']})
print(grouped_single)


grouped_multiple = dfminute.groupby(['srcIP', 'desIP']).size()
dfminute.groupby(['srcIP', 'desIP','packet']).size()
grouped_multiple = dfminute.groupby(['srpod', 'despod']).count()




##############################################################33
##  cinvert the categoriacal values to int 


dfDBaseSample=dfDBaseSample.drop(columns=['timestamp', 'IP'])
dfDBaseSample=dfDBaseSample.drop(columns=['converted_time' ])
dfDBaseSampleALLcol = handle_non_numerical_data(dfDBaseSample)
#pd.get_dummies(dfminute)
dfDBaseSampleALLcol.dtypes
 

hab =dfDBaseSampleALLcol[['srcrank','desrank','srpod','despod','packet', 'Hour','Minute']]
plt.figure(figsize=(20,5))
for i, x in enumerate(list(hab.columns)[:-1]):
  
    plt.subplot(1, 6, i+1)
    counts, bin_edges = np.histogram(hab[x], bins=10, density=True)
    print(x,':{}'.format(bin_edges))
    pdf = counts/sum(counts)
    print(x,':{}'.format(pdf))
    cdf = np.cumsum(pdf)
    print(x,':{}'.format(cdf))
    print('===========================')
    plt.plot(bin_edges[1:], pdf, bin_edges[1:], cdf)
    plt.xlabel(x)
    
  
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


sns.boxplot(x='srpod',y='Hour',data=dfDBaseSampleALLcol)
plt.show()

sns.boxplot(x='srpod',y='packet',data=dfDBaseSampleALLcol)
plt.show()

sns.boxplot(x='despod',y='Hour',data=dfDBaseSampleALLcol)
plt.show()


gr_intercluster =  dfDBaseSampleALLcol['intercluster']
gr_interdatacenter =  dfDBaseSampleALLcol['interdatacenter']
gr_srpod =  dfDBaseSampleALLcol['srpod']
gr_despod =  dfDBaseSampleALLcol['despod']
gr_packet =  dfDBaseSampleALLcol['packet']

grades_Hour =  dfDBaseSampleALLcol['Hour']

plt.scatter(grades_Hour, gr_srpod, color='r')
plt.scatter(gr_srpod, gr_despod, color='g')
plt.xlabel('Hour  Range')
plt.ylabel('Grades Scored')
plt.show()

plt.figure(figsize = (10,6))
sns.scatterplot(x='gr_srpod',y='gr_despod',color='r',data=dfDBaseSampleALLcol)
plt.title('Age vs Charges',size=18)
plt.xlabel('Age',size=14)
plt.ylabel('Charges',size=14)
plt.show()

cc = dfDBaseSampleALLcol[['srpod','despod']].corr()
print(cc)

cc = dfDBaseSampleALLcol[['srcrank','desrank']].corr()
print(cc)

cc = dfDBaseSampleALLcol[['intercluster','interdatacenter']].corr()
print(cc)
  
#################################################################
###             HADOOP 
#################################################################

df_all_list = []
##folders = glob('C:/Users/Filippos/Desktop/Master/CSC8634-Cloud/data/CLUSTER C')
folders = glob('C:/Users/Filippos/Desktop/Master/CSC8634-Cloud/data/CLUSTER C')
folders = [s for s in folders if "csv" not in s]

for j in folders:
   print('j',j)
   csv = glob(j + '/*')
   for i in csv:
       print('i',i)
       dfi = pd.read_csv(i,sep='\t' )
       dat=dfi
       dat.columns = [ 'timestamp','packet','srcIP','desIP','srcPort','desPort','IP','srchost','deshost', 'srcrank','desrank','srpod','despod','intercluster','interdatacenter']

       df_all_list.append(dat)
     
df_all_hadoop = pd.concat(df_all_list,axis=0)
df_all_hadoop.columns = ['timestamp','packet','srcIP','desIP','srcPort','desPort','IP','srchost','deshost', 'srcrank','desrank','srpod','despod','intercluster','interdatacenter']

 
converted_df = pd.to_datetime(df_all_hadoop['timestamp'], unit='s')
# Add new column to the DataFrame
dfHadoop = df_all_hadoop.assign(converted_time=converted_df)
print(dfHadoop)
dfHadoop.dtypes

#plot per momnth the records.
dfHadoop['Day']=dfHadoop.converted_time.dt.day
dfHadoop['Hour']=dfHadoop.converted_time.dt.hour
dfHadoop['Minute'] =dfHadoop.converted_time.dt.minute

days =[day for day, dfHadoop in dfHadoop.groupby('Day')]
daytrend = dfHadoop.groupby(['Day']).size()

fig = plt.figure(figsize = (10, 5))
# creating the bar plot
plt.bar(days, daytrend, color ='maroon', width = 0.2)
plt.xlabel("Days of a month")
plt.ylabel("No. of hits ")
plt.title("Wen inquiries per day ")
plt.show()

df2sampleHadoop = dfHadoop[(dfHadoop['Day'] == 1)]
df2sampleHadoop.groupby(['Day']).count()

hours =[hour for hour, df in dfHadoop[(dfHadoop['Day'] == 1)].groupby('Hour')]
#hours =[hour for hour, df in df2.groupby('Hour')]
#plt.plot(hours,df2[(df2['Day'] == 1)].groupby(['Hour']).size())                                         
plt.plot(hours,dfHadoop[(dfHadoop['Day'] == 1)].groupby(['Hour']).size())  
#df2.groupby(['Hour']).size()
plt.xticks(hours)
plt.grid()
plt.show()

############################# END  HADOOP  ########################

 

  
#############################################################
###  new with 4 variables 
gr_intercluster =  dfDBaseSample['intercluster']
gr_interdatacenter =  dfDBaseSample['interdatacenter']
gr_packet =  dfDBaseSample['packet']
grades_Hour =  dfDBaseSample['Hour']
gr_despod =  dfDBaseSample['despod']
gr_srpod =  dfDBaseSample['srpod']

 



fig, ax = plt.subplots(figsize=(10, 7))

# Titles
ax.set_title('SRC POD  vs. DES POD')
ax.set_xlabel('gr_srpod')
ax.set_ylabel('gr_despod')

# Remove top and right borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Adds major gridlines
ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.4)

#
scatter = ax.scatter(gr_srpod, gr_despod,
                     linewidths=1, alpha=0.75,
                     edgecolor='k',
                     s=[time  * time  for time  in grades_Hour],
                     c=gr_packet)

# Adds legend
kw = dict(prop="sizes",
          func=lambda s: np.sqrt(s),
          alpha=0.6)
legend1 = ax.legend(*scatter.legend_elements(**kw),
                    loc="upper left", title="Time",
                    labelspacing=2)
ax.add_artist(legend1)

handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
ax.legend(handles, labels, loc="upper right", title="Packet")

plt.tight_layout()
plt.savefig("plot.png")
plt.show()
## end 4 variable4s 
######################################################


#############################################################33
##  MODEL AND EVALUATION 
#############################################################


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
from sklearn.svm import SVC

 
 
dfDBaseSampleALLcol.fillna(0, inplace=True)
print(dfDBaseSampleALLcol.head())
dfDBaseSampleALLcol.dtypes

dfML = dfDBaseSampleALLcol[["packet","Hour","srpod","despod","srcrank","desrank","interdatacenter"]]

# Split-out validation dataset
array = dfML.values
X = array[:,0:6]
y = array[:,6]
 
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)


# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    
    
# Compare Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()



# Make predictions on validation dataset
#model = SVC(gamma='auto')
model =KNeighborsClassifier()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Evaluate predictions

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))  

print(Y_validation, predictions)

 # Creating a Confusion Matrix

cm = confusion_matrix(Y_validation, predictions)
df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (28,20))
fig, ax = plt.subplots()
sns.set(font_scale=1.4)
sns.heatmap(df_cm, annot=True, fmt='g')
class_names=[0,1]
tick_marks = np.arange(len(class_names))
plt.tight_layout()
plt.title('Confusion Matrix\n', y=1.1)
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
ax.xaxis.set_label_position("top")
plt.ylabel('Actual label\n')
plt.xlabel('Predicted label\n')


my_array = np.array([Y_validation, predictions])
print(my_array)


###########################END  #####################################

plt.figure(figsize=(10,10))
corr = dfDBaseSample.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True, annot=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
ax.set_yticklabels(
    ax.get_yticklabels(),
    rotation=45,

);


