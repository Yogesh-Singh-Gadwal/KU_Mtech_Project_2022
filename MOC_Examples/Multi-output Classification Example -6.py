"""

https://github.com/semihdesticioglu/twitter-nlp-classifier

"""

#!/usr/bin/env python
# coding: utf-8

"""
TRAIN CLASSIFIER
Disaster Resoponse Project
Udacity - Data Science Nanodegree

Arguments:
    1) SQLite db path (containing pre-processed data)
    2) pickle file name to save ML model
"""

# import libraries
import sys
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
import re
import nltk
nltk.download('stopwords')
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier


def load_data(filepath):
    """
    Load Data Function
    Arguments:
        filepath -> path to SQLite db
    Output:
        X -> input features
        Y -> target columns
        category_names -> for data visualization 
    """

    engine = create_engine("sqlite:///"+filepath)
    df = pd.read_sql_table("messages", con=engine)
    X = df["message"]
    Y = df.drop(columns=["message","id","original","genre"],axis=1)
    category_names = Y.columns
    return X, Y, category_names    


def tokenize(text):
    """
    Tokenize function
    
    Arguments:
        text -> list of twit messages 
    Output:
        clean_tokens -> cleaned tokenized twit texts
    """
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # url cleaner - replace with string "urlplaceholder"
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # get stopwords as a list
    stop_words = list(set(stopwords.words('english')))
    
    # lemmatize andremove stop words
    clean_tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens if word not in stop_words]
     
    return clean_tokens

def multiOutputF1_beta_score(y_true,y_pred,beta=2):
    """
    MultiOutput F1_beta_score
    
    This is a custom metric which is created with f1_beta. 
    
    It is used in gridsearch for scoring.
    
    I added extra weights for some critical disaster types.
    
    Since recall is very important to not miss important help issues, we choice beta as 2. 
    Check details for beta: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html
 
    """
    critical_types = ["search_and_rescue","missing_people","death","medical_products","medical_help","food","water"]
    score_sum = 0
    for column in range(0,y_true.shape[1]):
        score = fbeta_score(y_true.loc[:,y_true.columns[column]],y_pred[:,column], beta, average='binary')
        # multiply with 4 and changes the weights for some disaster types to predict them better with GridSearchCV
        if y_true.columns[column] in set(critical_types):
            score = score * 4
        score_sum += score
    avg_f1_beta = score_sum / ( y_true.shape[1] + (len(critical_types) * 4) - 7 )
    return  avg_f1_beta


def build_pipeline():
    """
    Build a pipeline and return a model object with GridSearch 
    
    This function creates a Pipeline which includes CountVectorizer, TfidfTransformer and a MultiOutput Classifier.
    
    Use pipeline to create GridSearch Brute search model. It needs training to find best parameters with Cross Validation.
    
    """
    pipeline =Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', MultiOutputClassifier(DecisionTreeClassifier())) ])
    
    # parameters searched previously and best parameters are determined as :
    parameters ={
                  'clf__estimator__min_samples_leaf': [30],
                   'clf__estimator__max_depth': [100]
                }
    # fbeta_score scoring object using make_scorer()
    scorer = make_scorer(multiOutputF1_beta_score,beta=2)
    
    grid_model = GridSearchCV(pipeline, param_grid=parameters, cv=3, scoring=scorer)
    
    return grid_model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate Model function
    
    This function evaluates model's performance with some metrics.
    Arguments:
        model -> Model with gridsearch by using pipeline
        X_test -> Test features
        Y_test -> Test labels
        category_names -> label names 
    """
    Y_pred = model.predict(X_test)
    
    # Print classification report on test data
    print('\n',classification_report(Y_test.values, Y_pred, target_names=category_names ))


def save_model(model, filepath):
    """ Saving model's best_estimator_ using pickle
    """
    pickle.dump(model.best_estimator_, open(filepath, 'wb'))


def main():
    ## This is the main function, if this script is called from another notebook or place, this main function will run only.
    ## This is "main" function's special feature.
    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
       
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_pipeline()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test,category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide correct arguments with correct paths. \n\nExample arguments: python '              'train_classifier.py DisasterResponse.db classifier.pkl ')
        
if __name__ == '__main__':
    main()
