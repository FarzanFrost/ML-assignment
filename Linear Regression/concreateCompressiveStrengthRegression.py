import pandas as pd

#Gathering data

location = "D:\\UCSC\\3rd year\\1st sem\\1. MLNC\\2. Assigment\\19001444\\data\\Linear Regression\\Concrete_Data.xls"
data = pd.read_excel( location )

#Data preprocessing is not required as all the attributes have real/integer values

x = data.iloc[ : , : 8 ]
y = data.iloc[ : , 8 ]

from sklearn.preprocessing import scale
x = scale( x )
y = scale( y )
#seperating train and test data

from sklearn.model_selection import train_test_split

xTrain , xTest , yTrain , yTest = train_test_split( x , y )

#Training linear regression model
from sklearn.linear_model import SGDRegressor

dataModel = SGDRegressor( max_iter = 5000 ).fit( xTrain , yTrain )

#Making prediction for test data set
predictedResults = dataModel.predict( xTest )

#Testing accuracy

meanAccuracy = dataModel.score( xTest , yTest )
print( meanAccuracy * 100 )

from sklearn.metrics import mean_squared_error
meanSquaredError = mean_squared_error( yTest , predictedResults )
print( meanSquaredError** (0.5) )