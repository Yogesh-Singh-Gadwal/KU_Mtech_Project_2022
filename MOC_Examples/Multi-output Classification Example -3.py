"""
https://github.com/prodo56/disaster-response-pipeline

"""

import sys
from sqlalchemy import create_engine
import pickle
import numpy as np
import pandas as pd
from utils import tokenize, ColumnSelector, Onehotencoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
# from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import joblib


def load_data(database_filepath):
    '''
    :param database_filepath: db path to load the data from
    
    :return: X,Y,category_names
    '''
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table("clean_data",engine)
    X = df[["message"]]
    Y = df.drop(["id","message","original","genre"],axis=1).values
    category_names = list(df.columns)[4:]
    
    return X,Y,category_names


def build_model():
    '''
    Build the pipeline with all the parameters needed for training
    
    :return cv: GridCV object
    '''
    pipeline = Pipeline([('text_processing', FeatureUnion([
        ('message_processing', Pipeline([
            ('select_messages', ColumnSelector(columns = ['message'])),
            ('vect', CountVectorizer(tokenizer = tokenize,max_features=5000)),
            ('tfidf', TfidfTransformer())
        ]))
    ])),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        'text_processing__message_processing__vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__n_estimators': [50, 100]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters,n_jobs=10,verbose=2)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evalute the trained model by looking at the classification report
    
    :param model: Trained model
    :param X_test: X_test dataset
    :param Y_test: Y_test labels
    :param category_names: category_names of all the messages in the test data
    '''
    Y_pred = model.predict(X_test)
    for column in range(Y_test.shape[1]):
        print("Column name: {}".format(category_names[column]))
        print(classification_report(Y_test[:,column],Y_pred[:,column],digits=6))


def save_model(model, model_filepath):
  # store the model as pickle object
    try:
        joblib.dump(model, model_filepath)
#         pickle.dump(model, open(model_filepath, 'wb'))
    except Exception as e:
        raise(e)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()