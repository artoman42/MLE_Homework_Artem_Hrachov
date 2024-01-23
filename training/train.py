"""
This script prepares the data, runs the training, and saves the model.
"""

import argparse
import os
import sys
import json
import logging
import pandas as pd
import time
import tensorflow as tf
from tensorflow.keras import layers, models
from datetime import datetime
from sklearn.model_selection import train_test_split

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = os.getenv('CONF_PATH') 

from utils import get_project_dir, configure_logging

# Loads configuration settings from JSON
CONF_FILE = "settings.json"
with open(CONF_FILE, "r") as file:
    conf = json.load(file)
# Defines paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train_file", 
                    help="Specify inference data file", 
                    default=conf['train']['table_name'])
parser.add_argument("--model_path", 
                    help="Specify the path for the output model")


class DataProcessor():
    def __init__(self) -> None:
        pass

    def prepare_data(self, max_rows: int = None) -> pd.DataFrame:
        logging.info("Preparing data for training...")
        df = self.data_extraction(TRAIN_PATH)
        df = self.data_rand_sampling(df, max_rows)
        return df

    def data_extraction(self, path: str) -> pd.DataFrame:
        logging.info(f"Loading data from {path}...")
        return pd.read_csv(path)
    
    def data_rand_sampling(self, df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
        if not max_rows or max_rows < 0:
            logging.info('Max_rows not defined. Skipping sampling.')
        elif len(df) < max_rows:
            logging.info('Size of dataframe is less than max_rows. Skipping sampling.')
        else:
            df = df.sample(n=max_rows, replace=False, random_state=conf['general']['random_state'])
            logging.info(f'Random sampling performed. Sample size: {max_rows}')
        return df


class Training():
    def __init__(self, input_dim, num_classes) -> None:
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = self.build_model()
    
    def build_model(self):
        logging.info("Building model...")
        model = models.Sequential()
        model.add(layers.Dense(64, activation="relu", input_dim = self.input_dim))
        model.add(layers.Dense(32, activation="relu"))
        model.add(layers.Dense(self.num_classes, activation="softmax"))
        return model
    def compile_model(self, optimizer=conf['train']['optimizer'],
                       loss=conf['train']['loss'], metrics=[conf['train']['metric']]):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def run_training(self, df: pd.DataFrame, out_path: str = None, test_size: float = 0.33) -> None:
        logging.info("Compiling model...")
        self.compile_model()
        logging.info("Running training...")
        X_train, X_test, y_train, y_test = self.data_split(df, test_size=test_size)
        start_time = time.time()
        self.train_model(X_train, y_train, X_test, y_test)
        end_time = time.time()
        logging.info(f"Training completed in {end_time - start_time} seconds.")
        self.test(X_test, y_test)
        self.save(out_path)

    def data_split(self, df: pd.DataFrame, test_size: float = 0.33) -> tuple:
        logging.info("Splitting data into training and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['target']), df['target'], test_size=test_size, 
                                random_state=conf['general']['random_state'])
        logging.info(f"Train dataset shape: ({X_train.shape[0]}, {X_train.shape[1] + 1}).")
        logging.info(f"Validation dataset shape: ({X_test.shape[0]},{X_test.shape[1] + 1})")
        logging.info("Transforming target to one-hot...")
        y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=self.num_classes)
        y_test_one_hot =  tf.keras.utils.to_categorical(y_test, num_classes=self.num_classes)
        return X_train, X_test, y_train_one_hot, y_test_one_hot
    
    def train_model(self, X_train, y_train, X_val, y_val,
                     epochs=conf["train"]["epochs"], batch_size=conf["train"]["batch_size"]):
        history = self.model.fit(X_train, y_train, 
                                 validation_data=(X_val, y_val),
                                 epochs=epochs, batch_size=batch_size)
        return history
    
    def test(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> float:
        logging.info("Testing the model...")
        loss, acc =self.model.evaluate(X_test, y_test) 
        logging.info(f"Validation Accuracy: {acc}, loss: {loss}")
        return acc, loss

    def save(self, path: str) -> None:
        logging.info("Saving the model...")
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        if not path:
            path = os.path.join(MODEL_DIR, datetime.now().strftime(conf['general']['datetime_format']) + '.keras')
        else:
            path = os.path.join(MODEL_DIR, path)
        logging.info(f"{path}")
        self.model.save(path)
        


def main():
    configure_logging()
    data_proc = DataProcessor()
    tr = Training(input_dim=conf['train']['input_dim'],
                  num_classes=conf['train']['num_classes'])

    df = data_proc.prepare_data(max_rows=conf['train']['data_sample'])
    tr.run_training(df, test_size=conf['train']['test_size'])


if __name__ == "__main__":
    main()