import sklearn
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

titanic_df = pd.read_csv('datasets/titanic_train.csv')
#Removing useless data
titanic_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Printing  all the null apears of each roll
# print(titanic_df[titanic_df.isnull().any(axis=1)].count())
#Result:
#Survived    179
#Pclass      179
#Sex         179
#Age           2
#SibSp       179
#Parch       179
#Fare        179
#Embarked    177

#Drop all data that is missing info
titanic_df = titanic_df.dropna()

#Quick Statistic overview
#print(titanic_df.describe())

#Data better visualization
#fig, ax = plt.subplots(figsize=(12,8))
#Testing many data visualization with plt
#plt.scatter(titanic_df['Age'], titanic_df['Survived'])
#plt.xlabel('Age')
#plt.ylabel('Survived')
#plt.scatter(titanic_df['Fare'], titanic_df['Survived'])
#plt.xlabel('Fare')
#plt.ylabel('Survived')

#Testing visualization with crossTab
#print(pd.crosstab(titanic_df['Sex'], titanic_df['Survived']))
#print(pd.crosstab(titanic_df['Pclass'], titanic_df['Survived']))

#Heatmap correlation
#titanic_data_corr = titanic_df.select_dtypes(include=[np.number]).corr() #Finds the correlation of the data and exclude non number variables(Male, female)
#fig,ax = plt.subplots(figsize=(12,10))
#sns.heatmap(titanic_data_corr, annot=True)
#plt.show()

label_enconding = preprocessing.LabelEncoder() #Convert categorical values to ordered integer values to use in ML algorithms

titanic_df['Sex'] = label_enconding.fit_transform(titanic_df['Sex'].astype(str)) #Convert sex into 0 or 1

titanic_df = pd.get_dummies(titanic_df, columns=['Embarked'], dtype=int)#Categorical values with no intrinsic order can be converted to numeric values using one-hot encoding, transform into false or true and separates into different columns

titanic_df = titanic_df.sample(frac=1).reset_index(drop=True) #Shufle the data

titanic_df.to_csv('datasets/titanic_processed_train.csv', index=False) #create the archive of the ready to use datasets
