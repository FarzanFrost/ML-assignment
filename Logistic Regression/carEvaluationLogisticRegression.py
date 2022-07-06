import pandas as pd

#Gathering data
location = "D:\\UCSC\\3rd year\\1st sem\\1. MLNC\\2. Assigment\\19001444\\data\\Logistic Regression\\car.data"
data = pd.read_csv( location )

#Preprocessing data

data[ 'Evaluation' ] = data[ 'Evaluation' ].eq( 'acc' ).astype( int )

buyingAndmaintenanceClassification = {
    
    'vhigh' : 0 ,
    'high' : 1 ,
    'med' : 2 ,
    'low' : 3
    
    }

data[ 'buying' ] = data[ 'buying' ].apply( lambda id : buyingAndmaintenanceClassification[ id ] )

data[ 'maintenance' ] = data[ 'maintenance' ].apply( lambda id : buyingAndmaintenanceClassification[ id ] )

doorsClassification = {
    
    '2' : 0 ,
    '3' : 1 ,
    '4' : 2 ,
    '5more' : 3
    
    }

data[ 'doors'] = data[ 'doors'].apply( lambda id : doorsClassification[ id ] )

personsClassification = {
    
    '2' : 0 ,
    '4' : 1 ,
    'more' : 2
    
    }

data[ 'persons' ] = data[ 'persons' ].apply( lambda id : personsClassification[ id ] )

luggageBootClassification = {
    
    'small' : 0 ,
    'med' : 1 ,
    'big' : 2
    
    }

data[ 'luggageBoot' ] = data[ 'luggageBoot' ].apply( lambda id : luggageBootClassification[ id ] )

safetyClassification = {
    
    'low' : 0 ,
    'med' : 1 ,
    'high' : 2
    
    }

data[ 'safety' ] = data[ 'safety' ].apply( lambda id : safetyClassification[ id ] )


x = data.iloc[ : , : 6 ]
y = data.iloc[ : , 6 ]


#seperating train and test data

from sklearn.model_selection import train_test_split

xTrain , xTest , yTrain , yTest = train_test_split( x , y )

#Training Logistic regression Model
from sklearn.linear_model import LogisticRegression

dataModel = LogisticRegression().fit( xTrain , yTrain )

#Predicting classes ( classification )
categorizedResults = dataModel.predict( xTest )

#Testing accuracy
meanAccuracy = dataModel.score( xTest , yTest )
print( meanAccuracy )
