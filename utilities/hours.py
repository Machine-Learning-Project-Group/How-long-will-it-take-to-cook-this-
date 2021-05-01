import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+')

h = ['hours', 'hour', 'hr', 'h', 'hrs']
s_dict = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
              "a": 1, "an": 1, "another": 1, "about": 1, "l": 1, "i": 1, "full": 1, "half": 0.5}

def standarize(value):
    if type(value)==int or value.isnumeric():
        return int(value)
    elif any(i.isnumeric() for i in value):
        return int([i for i in value if i.isnumeric()][-1])
    elif value in s_dict.keys():
        return int(s_dict[value])
    else:
        pass


def calc_hours(file):
    """
    calculate the specified time (in hours) spent in "steps", if any
    """
    data = pd.read_csv(file, header=0)
    
    # tokenize & lemmatize
    steps = data['steps'].apply(lambda steps: tokenizer.tokenize(steps)).apply(lambda steps: [lemmatizer.lemmatize(word, pos='n') for word in steps])
    
    # find index of those containing hour
    target = steps.apply(lambda x: any(i in h for i in x))
    tar_ind = target[target].index
    
    # extract the word/number before hour
    hours = steps[tar_ind].apply(lambda x: [x[i-1:i+1][0] for i in range(len(x)) if x[i] in h]).rename("hours")
    
    # standarize representation ('one' -> 1)
    # sum hours
    hours = hours.apply(lambda x: sum([standarize(i) for i in x if standarize(i)])).replace(0, np.NaN)
    
    # merge & save
    data = pd.merge(data, hours, how='left', right_on='hours', left_index=True, right_index=True)
    data.to_csv(file, header=True, index=False)


def mark_hours(train_file, test_file):
    """
    perform SVM on hour number to mark each instance,
    0 if predicted label = 1,2
    1 if predicted label = 3
    """
    train = pd.read_csv(train_file, header=0)
    test = pd.read_csv(test_file, header=0)
    
    
    
    train_hours = train.loc[:, ['duration_label', 'hours']].copy(deep=True).dropna()
    test_hours = test.loc[:, ['hours']].copy(deep=True).dropna()
    
    train_hours['hd'] = train_hours['hours'].apply(lambda x: x**2)
    test_hours['hd'] = test_hours['hours'].apply(lambda x: x**2)
                              
    # change label to binary
    binary = {1: 0, 2: 0, 3: 1}
    train_hours['duration_label'] = train_hours['duration_label'].apply(lambda x: binary[x])
    
    # prepare data
    X_train = train_hours.loc[:, ['hours', 'hd']]
    y_train = train_hours['duration_label']
    X_test = test_hours.loc[:, ['hours', 'hd']]
    
    # train & predict
    clf = svm.SVC(kernel='rbf', gamma='auto', C=1, probability=True)
    clf.fit(X_train, y_train)
    predict_train = clf.predict(X_train)
    predict_test = clf.predict(X_test)
    
    train_hours['hour_label'] = predict_train
    test_hours['hour_label'] = predict_test
    
    # merge
    train = pd.merge(train, train_hours['hour_label'], how='left', right_on = 'hour_label', left_index=True, right_index=True).fillna(int(0)).drop('hours', axis=1)
    test = pd.merge(test, test_hours['hour_label'], how='left', right_on = 'hour_label', left_index=True, right_index=True).fillna(int(0)).drop('hours', axis=1)
    
    train.to_csv(train_file, header=True, index=False)
    test.to_csv(test_file, header=True, index=False)