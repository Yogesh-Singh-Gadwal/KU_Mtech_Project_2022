"""
https://github.com/smabdullah/multioutput_classification/blob/master/multioutput_classification.py
Created on Mon Apr  8 15:45:47 2019

@author: SM Abdullah
@email: sma.csedu@gmail.com
"""

import numpy as np
import pandas as pd

# create a dummy dataset of 5000 elements and 294 features, having multioutput target variable
X = np.random.random((5000, 10))
Y = np.random.randint(2, size=(5000, 3))

# Split it for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# create a random forest model using n_estimators=300, max_features = 'sqrt' and random_state=0 for reproduction

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=300, max_features='sqrt', random_state=0)

# create a multioutput classifier

from sklearn.multioutput import MultiOutputClassifier

multi_target_classifier = MultiOutputClassifier(classifier, n_jobs=-1)

Y_pred = multi_target_classifier.fit(X_train, Y_train).predict(X_test)

# calculate multioutput precision and loss
from sklearn.metrics import label_ranking_average_precision_score, label_ranking_loss

avg_precision = label_ranking_average_precision_score(Y_test, Y_pred)
loss = label_ranking_loss(Y_test, Y_pred)

print('Average precision {0:.5f} and loss {1:.5f}'.format(avg_precision, loss))

# Convert the prediction into a Pandas dataframe and save it as a CSV file
Y_pred_df = pd.DataFrame(Y_pred)
# Y_pred_df.to_csv('Prediction.csv')
print(Y_pred_df.head())