"""
Script loads the latest trained model, data for inference and predicts results.
Imports necessary packages and modules.
"""
import time
import numpy as np
import argparse
import json
import logging
import os
import sys
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Model
import pandas as pd

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = os.getenv('CONF_PATH')
# CONF_FILE = "settings.json"
from utils import get_project_dir, configure_logging

# Loads configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Defines paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
RESULTS_DIR = get_project_dir(conf['general']['results_dir'])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--infer_file", 
                    help="Specify inference data file", 
                    default=conf['inference']['inp_table_name'])
parser.add_argument("--out_path", 
                    help="Specify the path to the output table")


def get_latest_model_path() -> str:
    """Gets the path of the latest saved model"""
    latest = None
    for (dirpath, dirnames, filenames) in os.walk(MODEL_DIR):
        for filename in filenames:
            if not latest or datetime.strptime(latest, conf['general']['datetime_format'] + '.keras') < \
                    datetime.strptime(filename, conf['general']['datetime_format'] + '.keras'):
                latest = filename
    return os.path.join(MODEL_DIR, latest)


def get_model_by_path(path: str) -> Model:
    """Loads and returns the specified model"""
    try:
        model = tf.keras.models.load_model(path)
        logging.info(f'Path of the model: {path}')
        return model
    except Exception as e:
        logging.error(f'An error occurred while loading the model: {e}')
        sys.exit(1)


def get_inference_data(path: str) -> pd.DataFrame:
    """loads and returns data for inference from the specified csv file"""
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logging.error(f"An error occurred while loading inference data: {e}")
        sys.exit(1)
        
def map_one_hot_to_descriptive_labels(one_hot_predictions, class_names=['0','1','2']):
    """Maps one-hot encoded predictions to descriptive class labels."""
    return [class_names[np.argmax(prediction)] for prediction in one_hot_predictions]

def predict_results(model: Model, infer_data: pd.DataFrame) -> pd.DataFrame:
    """Predict de results and join it with the infer_data"""
    logging.info(f"Inference data shape - {infer_data.shape}")
    start_time = time.time()
    results = model.predict(infer_data)
    end_time = time.time()
    logging.info(f"Inference completed in {end_time - start_time} seconds.")
    infer_data['results'] = map_one_hot_to_descriptive_labels(results)
    return infer_data


def store_results(results: pd.DataFrame, path: str = None) -> None:
    """Store the prediction results in 'results' directory with current datetime as a filename"""
    if not path:
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        path = datetime.now().strftime(conf['general']['datetime_format']) + '.csv'
        path = os.path.join(RESULTS_DIR, path)
    pd.DataFrame(results).to_csv(path, index=False)
    logging.info(f'Results saved to {path}')


def main():
    """Main function"""
    configure_logging()
    args = parser.parse_args()

    model = get_model_by_path(get_latest_model_path())
    infer_file = args.infer_file
    infer_data = get_inference_data(os.path.join(DATA_DIR, infer_file))
    results = predict_results(model, infer_data)
    store_results(results, args.out_path)

    logging.info(f'Prediction results: {results}')


if __name__ == "__main__":
    main()