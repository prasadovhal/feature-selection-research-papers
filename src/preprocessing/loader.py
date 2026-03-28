
import pandas as pd

def load_dataset(path: str):
    """Load dataset from CSV or Excel."""
    try:
        return pd.read_csv(path)
    except:
        return pd.read_excel(path)
