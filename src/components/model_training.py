import os
import sys
from src.exception import Custom_Exp
from src.logger import logging
from dataclasses import dataclass
from src.components.data_transformation import DataTransform, DataTransformConfig

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    AdaBoostRegressor, 
    GradientBoostingRegressor, 
    RandomForestRegressor
    )
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from sklearn.metrics import r2_score

from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainConfig:
    trained_model_file_path: str = os.path.join('artifact', 'model.pkl')

class ModelTrain:
    def __init__(self):
        self.model_train_config = ModelTrainConfig()

    def initiate_model_train(self, train_arr, test_arr):
        try:
            logging.info("Initializing input and output features")
            X_train, y_train, X_test, y_test = (
                train_arr[:, 1:],
                train_arr[:, 0],
                test_arr[:, 1:],
                test_arr[:, 0]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            logging.info("Initializing Evaluation Report")
            model_report = evaluate_model(X_train, y_train, X_test, y_test, models)

            logging.info("Selecting the best model")
            best_model_name = max(model_report, key = lambda x: model_report[x][1])

            best_model = models[best_model_name]

            r2_score = model_report[best_model_name][1]

            if r2_score < 0.6:
                raise Custom_Exp("No model found with r2 > 0.6")
            logging.info(f"Best model found as {best_model_name}")

            save_object(
                self.model_train_config.trained_model_file_path,
                best_model
            )

            return (best_model_name, r2_score)

        except Exception as e:
            logging.error("Fatal Error has occured in model training")
            raise Custom_Exp(e, sys)
        
