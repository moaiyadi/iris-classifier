# import argparse
# import os
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix

# def main(test_size: float, random_state: int):
#     # Load dataset
#     iris = load_iris()
#     X, y = iris.data, iris.target

#     # Train/test split
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=test_size, random_state=random_state
#     )

#     # Train model
#     model = DecisionTreeClassifier(random_state=random_state)
#     model.fit(X_train, y_train)

#     # Predictions
#     y_pred = model.predict(X_test)

#     # Accuracy
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"Accuracy: {accuracy:.4f}")

#     # Confusion matrix
#     cm = confusion_matrix(y_test, y_pred)

#     # Ensure output directory exists
#     output_dir = "D://UNI//AI//iris-classifier//outputs"
#     os.makedirs(output_dir, exist_ok=True)

#     # Save confusion matrix figure
#     plt.figure(figsize=(6, 5))
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
#                 xticklabels=iris.target_names,
#                 yticklabels=iris.target_names)
#     plt.xlabel("Predicted")
#     plt.ylabel("True")
#     plt.title("Confusion Matrix")
#     plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
#     plt.close()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Train Decision Tree on Iris dataset")
#     parser.add_argument("--test-size", type=float, default=0.2,
#                         help="Proportion of test data (default: 0.2)")
#     parser.add_argument("--random-state", type=int, default=42,
#                         help="Random state for reproducibility (default: 42)")

#     args = parser.parse_args()
#     main(args.test_size, args.random_state)

import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # for saving the model
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

    # Ensure output directory exists (relative path)
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Save trained model
    model_path = os.path.join(output_dir, "model.joblib")
    joblib.dump(model, model_path)

    # Save confusion matrix figure
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=iris.target_names,
                yticklabels=iris.target_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (Accuracy: {accuracy:.2f})")
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
