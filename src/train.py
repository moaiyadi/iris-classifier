# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier, plot_tree
# from sklearn.metrics import accuracy_score, confusion_matrix

# iris = load_iris()
# X = iris.data      # shape (150, 4)
# y = iris.target    # shape (150,)
# print(iris.feature_names, iris.target_names)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model = DecisionTreeClassifier(random_state=42)
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)
# print("Predictions:", y_pred[:5])
# print("True labels:", y_test[:5])

# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

# confusion = confusion_matrix(y_test, y_pred)
# print ("Confusion matrix:\n", 
#        f"{iris.target_names}\n", confusion)

# print(plot_tree(model))

# src/train.py

import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def main(test_size: float, random_state: int):
    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train model
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Ensure output directory exists
    output_dir = "D://UNI//AI//iris-classifier//outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Save confusion matrix figure
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=iris.target_names,
                yticklabels=iris.target_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Decision Tree on Iris dataset")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Proportion of test data (default: 0.2)")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random state for reproducibility (default: 42)")

    args = parser.parse_args()
    main(args.test_size, args.random_state)
