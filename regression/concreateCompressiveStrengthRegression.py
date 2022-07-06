import pandas as pd

#Gathering data

location = "D:\\UCSC\\3rd year\\1st sem\\1. MLNC\\2. Assigment\\19001444\\data\\regression\\Concrete_Data.xls"
data = pd.read_excel( location )

#Data preprocessing is not required as all the attributes have real/integer values

x = data.iloc[ : , : 8 ]
y = data.iloc[ : , 8 ]


#seperating train and test data

from sklearn.model_selection import train_test_split

xTrain , xTest , yTrain , yTest = train_test_split( x , y )

from sklearn.linear_model import LinearRegression

dataModel = LinearRegression().fit( xTrain , yTrain )

#Making prediction for test data set
predictedResults = dataModel.predict( xTest )

#Testing accuracy
from sklearn.metrics import mean_absolute_error
meanAbsoluteError = mean_absolute_error( yTest , predictedResults )
print( meanAbsoluteError )