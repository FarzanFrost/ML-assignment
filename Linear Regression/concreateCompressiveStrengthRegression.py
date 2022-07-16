import pandas as pd

#Gathering data

location = "D:\\UCSC\\3rd year\\1st sem\\1. MLNC\\2. Assigment\\19001444\\data\\Linear Regression\\Concrete_Data.xls"
data = pd.read_excel( location )

#Data preprocessing is not required as all the attributes have real/integer values

x = data.iloc[ : , : 8 ]
y = data.iloc[ : , 8 ]

# from sklearn.preprocessing import scale
# x = scale( x )
# y = scale( y )

#seperating train and test data

from sklearn.model_selection import train_test_split

xTrain , xTest , yTrain , yTest = train_test_split( x , y )

#Training linear regression model
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

dataModel = make_pipeline( StandardScaler() , SGDRegressor( alpha = 0.0001 ) ).fit( xTrain , yTrain )

#Making prediction for test data set
predictedResults = dataModel.predict( xTest )

#Testing accuracy
meanAccuracy = dataModel.score( xTest , yTest )
print( "Accuracy : " , meanAccuracy * 100 )
     
#Calculating mean squared error
from sklearn.metrics import mean_squared_error
meanSquaredError = mean_squared_error( yTest , predictedResults )
print( "Root mean squared error : " , meanSquaredError** ( 0.5 )  )

import matplotlib.pyplot as plt

xAxis = range( len( yTest ) )
plt.plot( xAxis , yTest , label="Original Data" )
plt.plot( xAxis , predictedResults , label="Predicted Data" )
plt.title( "Concrete compressive strength and predicted data" )
plt.xlabel( 'x-axis' )
plt.ylabel( 'y-axis' )
plt.legend( loc = 'best', fancybox = True , shadow = True )
plt.grid( True )
plt.show()
