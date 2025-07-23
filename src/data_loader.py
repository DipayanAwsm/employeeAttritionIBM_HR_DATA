import pandas as pd
from src.config import DATA_FILE_PATH

def load_data(filepath=DATA_FILE_PATH):
    df = pd.read_csv(filepath)
    return df
