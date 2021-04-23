import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def train_log(train_file, features):
    
    train = pd.read_csv(train_file, header=0)

    # double check given features
    features = [f for f in features if f in train.columns]
    print(f"Training logistic regression on: {features}\n")

    # extract features
    X_train = train.loc[:, features]
    y_train = train['duration_label']

    # train
    model = LogisticRegression(random_state=42, max_iter=2000).fit(X_train, y_train)
    return model

def predict_log(test_file, features, model, name='log_prediction'):
    
    # extract features
    test = pd.read_csv(test_file, header=0)
    X_test = test.loc[:, features]
    
    # predict
    test[name] = model.predict(X_test)
    
    # save
    print(f"Predicted {test_file}, saved to column '{name}'\n")
    test.to_csv(test_file, index=False)