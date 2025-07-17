import sys
import os
from src.exception import Custom_Exp
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

@dataclass
class DataTransformConfig:
    preprocessor_obj_file_path = os.path.join('artifact', 'preprocessor.pkl')


class DataTransform:
    def __init__(self):
        self.preprocessing_config = DataTransformConfig()

    def get_data_transformer(self):
        """
        This function is responsible for transforming the data
        """
        try:
            cat_features = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']
            num_features = ['reading_score','writing_score']

            logging.info("Initializing Pipelines")
            num_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy = 'median')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(drop = 'first'))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("Numerical Pipeline", num_pipeline, num_features),
                    ("Categorical Pipeline", cat_pipeline, cat_features)
                ]
            )
            logging.info("Pipelines Initialized")

            return preprocessor
        
        except Exception as e:
            logging.error("Fatal Error has occured in creating pipelines")
            raise Custom_Exp(e, sys)
        
    def initiate_data_transform(self, train_path, test_path):
        try:
            logging.info("Initializing test and train dataframes")
            train = pd.read_csv(train_path)
            test = pd.read_csv(test_path)
            logging.info("test and train dataframes initialized")

            preprocessing_obj = self.get_data_transformer()
            logging.info('Preprocessing pipeline loaded')

            target_column_name = 'math_score'

            X_train = train.drop(target_column_name, axis = 1)
            y_train = train[target_column_name]

            X_test = test.drop(target_column_name, axis = 1)
            y_test = test[target_column_name]
            logging.info("X_test, X_train, y_train, y_test initialized")

            logging.info("Scaling X_train and X_test")
            X_train_scaled = preprocessing_obj.fit_transform(X_train)
            X_test_scaled = preprocessing_obj.transform(X_test)
            logging.info("Successfully scaled X_train, X_test")

            train_arr = np.c_[y_train.to_numpy(), X_train_scaled]
            test_arr = np.c_[y_test.to_numpy(), X_test_scaled]
            logging.info("Converting and concatinating train and test arrays")
            
            logging.info("Saving preprocessing file")

            save_object(
                file_path = self.preprocessing_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.preprocessing_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.error('Fata error has occured in transforming data')
            raise Custom_Exp(e, sys)