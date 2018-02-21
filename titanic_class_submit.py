from pandas import DataFrame
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

#Importing Data
train_set = pd.read_csv('train.csv')
test_set = pd.read_csv('test.csv')
clean_set = [train_set, test_set]

#Data Clearning
label = LabelEncoder()
for dataset in clean_set:
	dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)
	dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)
	dataset['Embarked'] = label.fit_transform(dataset['Embarked'])
	dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
	dataset['Title'] = dataset['Title'].map({'Mr':0, 'Mrs': 1, 'Miss': 2, 'Master': 3})
	dataset['Title'] = dataset['Title'].fillna(4)
	dataset['Sex'] = label.fit_transform(dataset['Sex'])

features = train_set[['Pclass','Sex','SibSp','Parch','Fare','Embarked','Title']]
survival = train_set[['Survived']]

test_X = test_set[['Pclass','Sex','SibSp','Parch','Fare','Embarked','Title']]

#Training Model
XGB_Model = XGBClassifier()
XGB_Model.fit(features, survival.values.ravel())

#Testing Model and Submitting
XGBy_pred = XGB_Model.predict(test_X) #Predicting based on testing data
submit = [test_set['PassengerId'], XGBy_pred]
submit = DataFrame(submit, index = ['PassengerId', 'Survived']).T
submit = submit.set_index('PassengerId')
submit.to_csv('predictions.csv')