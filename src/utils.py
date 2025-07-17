import os
import sys
import pickle
from src.exception import Custom_Exp
from src.logger import logging
from sklearn.metrics import r2_score
import numpy as np

def save_object(file_path, obj):
    try:
        path = os.path.dirname(file_path)
        os.makedirs(path, exist_ok = True)

        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)

    except Exception as e:
        logging.error(f"Fatal error has occured while saving object to {file_path}")
        raise Custom_Exp(e, sys)

def adjusted_r2(X, y_actual, y_predicted):
    score = r2_score(y_actual, y_predicted)
    adj_score = 1 - ((1 - score) * (X.shape[0] - 1) / (X.shape[0] - X.shape[1] - 1))
    return adj_score

def evaluate_model(X_train, y_train, X_test, y_test, models: dict) -> dict:
    try:
        report = {}
        logging.info("Looping over all models and evaluating r2")
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]
            model.fit(X_train, y_train)

            #For Training r2
            logging.info(f"Evaluating training r2 for model {model_name}")
            y_train_pred = model.predict(X_train)
            y_train_score = adjusted_r2(X_train, y_train, y_train_pred)

            #For test r2
            logging.info(f"Evaluating testing r2 for model {model_name}")
            y_test_pred = model.predict(X_test)
            y_test_score = adjusted_r2(X_test, y_test, y_test_pred)
            
            #Evaluating 
            report[model_name] = [y_train_score, y_test_score]

        return report

    except Exception as e:
        logging.error("Fata error has occured in model evaluation")