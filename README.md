# Math Score Prediction API
A Flask application for predicting mathematics scores among students. This can help aid in identifying students in need of supplementary materials and or additional guidance.

## Key Features
- Predicts Math Scores on exam 
- Identifies the best Machine Learning Algorithm for the data
- RESTful API endpoints using Flask
- Complete Data Ingestion, Data Transformation, and Model Training modules

## Installation
1. Clone the repository
- git clone https://github.com/Sebinate/MLProject-1.git
- cd MLProject-1

2. Create Conda Environment
- conda create -p venv python==3.11.9 -y
- conda init powershell (Used for ease, can be removed)
- conda activate venv/
- code . (Opens Visual Studio Code)
- pip install -r requirements

## Usage
**NOTE: Ensure that you are in the MLProject-1/ folder**
- python app.py

## Project Structure
- 'setup.py'
- 'app.py' - Main Flask application
- 'requirements.txt' - List of dependencies
- 'templates/' - Directory for Flask templates
- 'src/'
    - 'components/' - Directory for data ingestion, transformation, and model training
    - 'pipeline/' - Directory for new predictions
    - 'logger.py' - Custom logging
    - 'exception.py' - Custom exception handling
    - 'utils.py' -  Functions used across the project
- 'notebook/' - Contain EDA, initial model training, and data used for the projects
