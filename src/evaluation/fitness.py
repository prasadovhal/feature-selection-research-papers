
from sklearn.metrics import accuracy_score

def fitness(y_true, y_pred, n_features):
    """Fitness = accuracy penalized by number of features."""
    return accuracy_score(y_true, y_pred) / (1 + 0.01 * n_features)
