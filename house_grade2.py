# -*- coding: utf-8 -*-
"""
Created on Sat May  9 19:49:30 2020

@author: Surraj
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

test = pd.read_csv('test.csv')
df = pd.read_csv('train.csv')

msno.matrix(df,figsize=(12,5))
msno.matrix(test,figsize=(12,5))


def null(df):
    null_value = df.isnull().sum()
    per_null = 100*df.isnull().sum()/len(df)
    
    unique=pd.DataFrame(columns=['unique'])
    for i in df.columns:
        nunique = df[i].nunique()
        unique.loc[i]=[nunique]
        
    miss_val = pd.concat([null_value,per_null,unique],1)
        
    miss_val_rename = miss_val.rename(columns = {0: 'Missing Values',
                                                 1: '% of missing values',
                                                 2: 'Unique Values'})
    miss_val_rename = miss_val_rename.sort_values('% of missing values',ascending = False)
    
    return  miss_val_rename

null(df)

def ree(x):
    
    if x['Grade'] == 'A':
        return '1'
    elif x['Grade'] == 'B':
        return '2'
    elif x['Grade'] == 'C':
        return '3'
    elif x['Grade'] == 'D':
        return '4'
    else:
        return '5'
df['Grade'] = df.apply(lambda  x:ree(x),axis=1)

fig, ax = plt.subplots(nrows=3, ncols=3)
fig.set_size_inches(12,10)

sns.boxplot(data=df, x='roof',y='Grade',ax=ax[0][0])
sns.boxplot(data=df, x='Troom',y='Grade',ax=ax[0][1])
sns.boxplot(data=df, x='Nbedrooms',y='Grade',ax=ax[0][2])
sns.boxplot(data=df, x='Nbwashrooms',y='Grade',ax=ax[1][0])
sns.boxplot(data=df, x='Twashrooms', y='Grade',ax=ax[1][1])
sns.boxplot(data=df, x='Nfloors',y='Grade',ax=ax[1][2])
#sns.boxplot(data=df, x='Grade', y='EXPECTED',ax=ax[2][0])
sns.boxplot(data=df, x='ANB', y='Grade',ax=ax[2][0])

fig, ax = plt.subplots(nrows=2, ncols=2)
fig.set_size_inches(12,10)

sns.barplot(data=df, x='roof',y='EXPECTED',hue='Grade',ax=ax[0][0])
sns.barplot(data=df, x='Troom',y='EXPECTED',hue='Grade',ax=ax[0][1])
#sns.barplot(data=df, x='Nbedrooms',y='EXPECTED',hue='Grade',ax=ax[0][2])
sns.barplot(data=df, x='Nbwashrooms',y='EXPECTED',hue='Grade',ax=ax[1][0])
sns.barplot(data=df, x='Twashrooms', y='EXPECTED',hue='Grade',ax=ax[1][1])
#sns.barplot(data=df, x='Nfloors',y='EXPECTED',hue='Grade',ax=ax[1][2])
#sns.boxplot(data=df, x='Grade', y='EXPECTED',ax=ax[2][0])
#sns.barplot(data=df, x='ANB', y='EXPECTED',hue='Grade',ax=ax[2][0])

df['Grade'] = df['Grade'].astype('int64')
sns.countplot(df['Grade'])


fig,ax = plt.subplots()
fig.set_size_inches(20,10)
sns.heatmap(df.corr(),vmax=.8,annot=True,square=True)


fig, ax = plt.subplots(nrows=2, ncols=2)
fig.set_size_inches(12,10)

sns.distplot(df['Roof(Area)'],ax=ax[0][0])
sns.distplot(df['Lawn(Area)'],ax=ax[0][1])
sns.distplot(df['Grade'],ax=ax[1][0])
sns.distplot(df['EXPECTED'],ax=ax[1][1])


fig,ax = plt.subplots(nrows=3)
fig.set_size_inches(12,10)

orderr = [2,3,4,5,6,7,8]
aggre = pd.DataFrame(df.groupby(['Troom','Nbedrooms'])['Grade'].mean()).reset_index()
sns.barplot(data=aggre,x='Nbedrooms',y='Grade',ax=ax[0])

aggre = pd.DataFrame(df.groupby(['Troom','Nbedrooms'])['Grade'].mean()).reset_index()
sns.pointplot(data=aggre,x='Nbedrooms',y='Grade',hue='Troom',ax=ax[1],order=orderr)

aggre = pd.DataFrame(df.groupby(['Nbwashrooms','Nbedrooms'])['Grade'].mean()).reset_index()
sns.pointplot(data=aggre,x='Nbedrooms',y='Grade',hue='Nbwashrooms',ax=ax[2],order=orderr)

del aggre,ax,orderr

df['Roof(Area)'] = df['Roof(Area)'].fillna(df['Roof(Area)'].mean())
df['roof'] = df['roof'].fillna('Unknown')

test['Roof(Area)'] = test['Roof(Area)'].fillna(test['Roof(Area)'].mean())
test['roof'] = test['roof'].fillna('Unknown')

df['roof'] = df.roof.apply(lambda x:str(x).upper())
test['roof'] = test.roof.apply(lambda x:str(x).upper())

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df['Grade'] = lb.fit_transform((df['Grade']))

test['EXPECTED'] = test['EXPECTED'].apply(lambda x: str(x).split('$')[0]).astype('int')

cool = ['roof']
df = pd.get_dummies(df,columns=cool)
test = pd.get_dummies(test,columns=cool)

y = df.Grade
X = df.drop(['Id','Twashrooms','Grade'],1)
test = test.drop(['ID','Twashrooms'],1)
'''
colnames = X.columns

X = X.fillna(X.mean())


from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.2)
lasso_coef = lasso.fit(X,y)

# Compute and print the coefficients
lasso_coef = lasso.coef_
print(lasso_coef)


# Plot the coefficients
plt.plot(range(len(colnames)), lasso_coef)
plt.xticks(range(len(colnames)), colnames.values, rotation=60) 
plt.margins(0.02)
plt.show()'''

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
# Create a linear regression object: reg
reg = LinearRegression()
ref = RandomForestClassifier()
# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv=5)
cv_scores = cross_val_score(ref, X, y, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

# find the mean of our cv scores here
print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

predict = ref.predict(test)






