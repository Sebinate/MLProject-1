import os
import sys
import pandas as pd
from src.exception import Custom_Exp
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:

            model_path = 'artifact/model.pkl'
            preprocessor_path = 'artifact/preprocessor.pkl'

            logging.info("Loading Model and Proprocesser")
            model = load_object(
                model_path
            )

            preprocessor = load_object(
                preprocessor_path
            )
            logging.info("Loaded Model and Proprocesser")
            data_scaled = preprocessor.transform(features)

            prediction = model.predict(data_scaled)
            logging.info("Successfully Scaled and Predicted data")

            return prediction
        
        except Exception as e:
            logging.error("Fata Error has occured in reading inputted data")
            raise Custom_Exp(e, sys)

class CustomData:
    def __init__(
            self,
            gender: str,
            race_ethnicity: str,
            parental_level_of_education: str,
            lunch: str,
            test_preparation_course: str,
            reading_score: int,
            writing_score: int
        ):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = int(reading_score)
        self.writing_score = int(writing_score)

    def get_data_as_frame(self):
        try:
            custom_data_input = {
                'gender': [self.gender],
                'race_ethnicity': [self.race_ethnicity],
                'parental_level_of_education': [self.parental_level_of_education],
                'lunch': [self.lunch],
                'test_preparation_course': [self.test_preparation_course],
                'reading_score': [self.reading_score],
                'writing_score': [self.writing_score]
            }

            return pd.DataFrame(custom_data_input)

        except Exception as e:
            logging.error("Fata Error in converting web data to pandas dataframe")
            raise Custom_Exp(e, sys)
