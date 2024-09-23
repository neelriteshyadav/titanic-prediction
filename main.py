
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 13:23:49 2023

@author: neelriteshyadav
"""

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Model Implementation
# 1. Support Vector Machine (SVM):
def svm_method(X_train_scaled, y_train, X_test_scaled):
    # Use GridSearchCV to find the best hyperparameters
    param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'kernel': ['linear', 'rbf', 'poly']}
    grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    # Get the best parameters and predict with the best estimator
    best_params = grid_search.best_params_
    print(f"Best SVM Parameters: {best_params}")
    
    best_svm_model = grid_search.best_estimator_
    svm_predictions = best_svm_model.predict(X_test_scaled)
    return svm_predictions

# 2. Neural Network (MLP):
def mlp_method(X_train_scaled, y_train, X_test_scaled):
    # Use GridSearchCV to find the best hyperparameters
    param_grid = {'hidden_layer_sizes': [(50,), (100,), (150,)], 'max_iter': [1000, 1500, 2000], 'activation': ['logistic', 'tanh', 'relu']}
    grid_search = GridSearchCV(MLPClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    # Get the best parameters and predict with the best estimator
    best_params = grid_search.best_params_
    print(f"Best MLP Parameters: {best_params}")
    
    best_mlp_model = grid_search.best_estimator_
    mlp_predictions = best_mlp_model.predict(X_test_scaled)
    return mlp_predictions


# 3. Random Forest:
def rf_method(X_train, y_train, X_test):
    # Use GridSearchCV to find the best hyperparameters
    param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters and predict with the best estimator
    best_params = grid_search.best_params_
    print(f"Best Random Forest Parameters: {best_params}")
    
    best_rf_model = grid_search.best_estimator_
    rf_predictions = best_rf_model.predict(X_test)
    return rf_predictions

# Performance Metrics
def evaluate(predictions, y_test):
    accuracy = accuracy_score(y_test, predictions) * 100
    confusion = confusion_matrix(y_test, predictions)
    return accuracy, confusion

# Print results
def print_evaluation(method_name, accuracy, confusion):
    print(f"{method_name} Accuracy: {accuracy}")
    print(f"{method_name} True Positives (TP): {confusion[1, 1]} (Survived and correctly predicted)")
    print(f"{method_name} True Negatives (TN): {confusion[0, 0]} (Not Survived and correctly predicted)")
    print(f"{method_name} False Positives (FP): {confusion[0, 1]} (Not Survived but incorrectly predicted as Survived)")
    print(f"{method_name} False Negatives (FN): {confusion[1, 0]} (Survived but incorrectly predicted as Not Survived)\n")

# Plot confusion matrix
def plot_confusion_matrix(conf_matrix, method_name):
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{method_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# Plot accuracy comparison
def plot_accuracy_comparison(accuracies):
    methods = ["SVM", "MLP", "Random Forest"]
    plt.bar(methods, accuracies, color=['blue', 'orange', 'green'])
    plt.title("Accuracy Comparison of Different Methods")
    plt.xlabel("Methods")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.show()

def main():
    # Data Preprocessing
    data_path = "/Users/neelriteshyadav/Downloads/589/train.csv"
    df = pd.read_csv(data_path)

    # Drop unnecessary columns or handle missing values
    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)

    # Convert categorical variables to numerical
    df = pd.get_dummies(df, columns=["Sex", "Embarked", "Pclass"], drop_first=True)

    # Split data into features and target
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model Implementation and Evaluation
    svm_predictions = svm_method(X_train_scaled, y_train, X_test_scaled)
    mlp_predictions = mlp_method(X_train_scaled, y_train, X_test_scaled)
    rf_predictions = rf_method(X_train, y_train, X_test)

    # Performance Metrics
    accuracies = []
    svm_accuracy, svm_matrix = evaluate(svm_predictions, y_test)
    accuracies.append(svm_accuracy)
    print_evaluation("SVM", svm_accuracy, svm_matrix)
    plot_confusion_matrix(svm_matrix, "SVM")

    mlp_accuracy, mlp_matrix = evaluate(mlp_predictions, y_test)
    accuracies.append(mlp_accuracy)
    print_evaluation("MLP", mlp_accuracy, mlp_matrix)
    plot_confusion_matrix(mlp_matrix, "MLP")

    rf_accuracy, rf_matrix = evaluate(rf_predictions, y_test)
    accuracies.append(rf_accuracy)
    print_evaluation("Random Forest", rf_accuracy, rf_matrix)
    plot_confusion_matrix(rf_matrix, "Random Forest")

    # Plot accuracy comparison
    plot_accuracy_comparison(accuracies)

if __name__ == "__main__":
    main()
