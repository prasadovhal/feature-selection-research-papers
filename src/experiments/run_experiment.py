from src.preprocessing.loader import load_dataset
from src.preprocessing.preprocess import preprocess_dataframe


def main() -> None:
    """Run a minimal preprocessing smoke-test experiment."""
    data = load_dataset("datasets/biodeg.csv")
    x_features, y_target = preprocess_dataframe(data)
    print("Dataset shape:", x_features.shape, "Target shape:", y_target.shape)

if __name__ == "__main__":
    main()
