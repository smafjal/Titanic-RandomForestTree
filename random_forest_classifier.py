# smafjal
import numpy as np
import pandas as pd
import pickle

import xgboost
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV,KFold,train_test_split

import data_preprocessing as data_loader


def validate_model_kfold(clf,data_X,data_Y):
    kf = KFold(n_splits=10)
    outcomes = []
    for fold,(train_index, test_index) in enumerate(kf.split(data_X,data_Y)):
        train_X, test_X = data_X.values[train_index], data_X.values[test_index]
        train_Y, test_Y = data_Y.values[train_index], data_Y.values[test_index]
        clf.fit(train_X, train_Y)
        predictions = clf.predict(test_X)
        accuracy = accuracy_score(test_Y, predictions)
        outcomes.append(accuracy)
        print("Fold {0} accuracy: {1}".format(fold, accuracy))

    mean_outcome = np.mean(outcomes)
    print("Mean Accuracy -->: {0}".format(mean_outcome))
    print ("*"*50)

def train_random_forest_classifier(train_X,train_Y,test_X,test_Y):
    clf = RandomForestClassifier()
    parameters = {'n_estimators': [4, 6, 9],
                  'max_features': ['log2', 'sqrt','auto'],
                  'criterion': ['entropy', 'gini'],
                  'max_depth': [2, 3, 5, 10],
                  'min_samples_split': [2, 3, 5],
                  'min_samples_leaf': [1,5,8]}

    # Type of scoring used to compare parameter combinations
    acc_scorer = make_scorer(accuracy_score)

    # Run the grid search
    grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
    grid_obj = grid_obj.fit(train_X, train_Y)

    # Set the clf to the best combination of parameters
    clf = grid_obj.best_estimator_

    # Fit the best algorithm to the data.
    clf.fit(train_X,train_Y)
    predictions = clf.predict(test_X)
    print(accuracy_score(test_Y, predictions))
    return clf

def generate_submission(clf,data_test):
    passenger_id = data_test['PassengerId']
    predictions = clf.predict(data_test.drop('PassengerId', axis=1))
    output = pd.DataFrame({ 'PassengerId' : passenger_id, 'Survived': predictions })
    output.to_csv('model/rft-predicted-value-titanic.csv', index = False)
    print output.head()


def get_label_data(data):
    data_Y=data['Survived']
    data_X = data.drop(['Survived', 'PassengerId'],axis=1)
    return data_X,data_Y

def data_train_test_split(data_X,data_Y):
    num_test = 0.20
    train_X,test_X,train_Y,test_Y = train_test_split(data_X,data_Y, test_size=num_test, random_state=23)
    return train_X,train_Y,test_X,test_Y

def save_model(model,path):
    with open(path,'wb') as file:
        pickle.dump(model,file)

def load_model(path):
    with open(path,'rb') as file:
        model=pickle.load(file)
        return model

def main(mode='train'):
    train_path='data/train.csv'
    test_path='data/test.csv'

    data_train=data_loader.preprocess_data(train_path)
    data_test=data_loader.preprocess_data(test_path)
    data_X,data_Y = get_label_data(data_train)

    if mode in 'train':
        train_X, train_Y,test_X,test_Y = data_train_test_split(data_X,data_Y)
        print ("Model in traning........")
        clf=train_random_forest_classifier(train_X=train_X,train_Y=train_Y,test_X=test_X,test_Y=test_Y)
        save_model(clf,'model/rft-model.pkl')
        del clf

    clf=load_model('model/rft-model.pkl')
    print ('\nValidation of model(kfold)')
    validate_model_kfold(clf,data_X,data_Y)

    print ("Write predicted value to disk")
    generate_submission(clf,data_test=data_test)


if __name__ == '__main__':
    main('train')
