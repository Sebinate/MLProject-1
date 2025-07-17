from flask import Flask, request, render_template

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import Custom_Exp
from src.logger import logging

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['POST', 'GET'])
def predition():
    if request.method == "POST":
        data = CustomData(
            gender = request.form.get('gender'),
            parental_level_of_education = request.form.get('parental_level_of_education'),
            lunch = request.form.get('lunch'),
            test_preparation_course = request.form.get('test_preparation_score'),
            race_ethnicity = request.form.get('race'),
            reading_score = request.form.get('reading_score'),
            writing_score = request.form.get('writing_score'),
        )

        pred_df = data.get_data_as_frame()

        predict_pipeline = PredictPipeline()
        result = round(predict_pipeline.predict(pred_df)[0], 2)
        return render_template('predict.html', results = result)

    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(debug = True, host = '0.0.0.0', port = 5000)