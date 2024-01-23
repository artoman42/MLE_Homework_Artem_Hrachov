import unittest
import pandas as pd
import os
import sys
import json

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = os.getenv('CONF_PATH')

from training.train import DataProcessor, Training 


class TestDataProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(CONF_FILE, "r") as file:
            conf = json.load(file)
        cls.data_dir = conf['general']['data_dir']
        cls.train_path = os.path.join(cls.data_dir, conf['train']['table_name'])

    def test_data_extraction(self):
        dp = DataProcessor()
        df = dp.data_extraction(self.train_path)
        self.assertIsInstance(df, pd.DataFrame)

    def test_prepare_data(self):
        dp = DataProcessor()
        df = dp.prepare_data(150)
        self.assertEqual(df.shape[0], 150)


class TestTraining(unittest.TestCase):
    def test_train(self):
        tr = Training()
        
        X_train = pd.DataFrame({
            'sepal length (cm)': [5.1, 4.9, 4.7],
            'sepal width (cm)': [3.5, 3.0, 3.2],
            'petal length (cm)':[1.4, 1.4, 1.3],
            'petal width (cm)':[0.2, 0.2, 0.2]
        })
        y_train = pd.Series([0, 1, 2])
        tr.train_model(X_train, y_train)
        self.assertIsNotNone(tr.model)


if __name__ == '__main__':
    unittest.main()