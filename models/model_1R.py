import numpy as np
import pandas as pd

def train_1R(train_file, features):
    """Trains on the given csv file, based on the given features
    Returns the learned 1R model in the form dict of dict,
    feature: prediction_rule"""
    
    train = pd.read_csv(train_file, header=0)

    # double check given features
    features = [f for f in features if f in train.columns]
    print(f"Training 1R model on: {features}\n")
    
    y = train['duration_label']
    model = {}
    
    for feature in features:
        
        X = train[feature]
        
        # count frequency of each n_steps in each duration label
        matrix = pd.DataFrame(index=set(X), columns=[1, 2, 3]).fillna(0)
        for i in range(len(X)):
            matrix.loc[X[i], y[i]] += 1

        # calculate the most frequent label in each n_step, save to list
        predict = {}
        for index, a in matrix.iterrows():

            predict[index] = [i for i, j in enumerate(a) if j == max(a)][0] + 1

        # save prediction to dictionary
        model[feature] = predict

    return model

def predict_1R(test_file, model):
    """Predicts the labels from the given file & 1R model"""
    
    test = pd.read_csv(test_file, header=0)
    
    for feature in model.keys():
        
        if feature in test.columns:
            
            prediction = []
            for index, row in test.iterrows():

                X = row[feature]

                if X in model[feature].keys():
                    prediction.append(model[feature][X])

                else:
                    prediction.append(2)

            test[(feature+"_1R_prediction")] = prediction
        
        else:
            print(f"Feature {feature} not found in the test file!")
    
    print(f"Predicted {test_file} with {list(model.keys())}\n")
    test.to_csv(test_file, index=False)