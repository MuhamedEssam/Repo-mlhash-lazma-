import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
dataset = pd.read_csv('Customer_Churn_Modelling.csv')
X = dataset.drop(labels=['CustomerId', 'Surname', 'RowNumber', 'Exited'], axis = 1)
y = dataset['Exited']

label1 = LabelEncoder()
X['Geography'] = label1.fit_transform(X['Geography'])
X = pd.get_dummies(X, drop_first=True, columns=['Geography'])
label = LabelEncoder()
X['Gender'] = label.fit_transform(X['Gender'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# load and evaluate a saved model

from keras.models import load_model
 
# load model
model = load_model('model.h5')
# summarize model.
model.summary()

# evaluate the model
score = model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))