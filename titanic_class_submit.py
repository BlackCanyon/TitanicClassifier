from pandas import DataFrame
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import sklearn
from sklearn.model_selection import train_test_split

def agerange(x):
	if 'Master' in x:
		x = 0
	else:
		x = 1
	return x

#Importing Data
train_set = pd.read_csv('train.csv')
train_set['Sex'] = train_set['Sex'].map({'female': 1, 'male': 0})
test_set = pd.read_csv('test.csv')
test_set['Sex'] = test_set['Sex'].map({'female': 1, 'male': 0})

#Training Data Clearning
features = train_set[['Pclass','Sex', 'SibSp', 'Parch', 'Name']]
features['Name'] = features['Name'].map(agerange)
survival = train_set[['Survived']]

#Test Data Cleaning
test_X = test_set[['Pclass', 'Sex', 'SibSp', 'Parch', 'Name']]
test_X['Name'] = test_X['Name'].map(agerange)

#Training Model
RF_Model = GradientBoostingClassifier()
RF_Model.fit(features, survival.values.ravel())

#Testing Model and Submitting
RFy_pred = RF_Model.predict(test_X) #Predicting based on testing data
submit = [test_set['PassengerId'], RFy_pred]
submit = DataFrame(submit, index = ['PassengerId', 'Survived']).T
submit = submit.set_index('PassengerId')
submit.to_csv('predictions.csv')