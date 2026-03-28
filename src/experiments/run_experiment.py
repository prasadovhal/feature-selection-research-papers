
from src.preprocessing.loader import load_dataset
from src.preprocessing.preprocess import preprocess

def main():
    data = load_dataset("data/raw/sample.csv")
    X, y = preprocess(data)
    print("Dataset shape:", X.shape)

if __name__ == "__main__":
    main()
