from typing import Optional
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

import logging
import pandas as pd


def clean_data(games: pd.DataFrame) -> Optional[str]:
    logger = logging.getLogger(__name__)
    tbd_cleaned_games = games.drop(games[games['user_score'] == 'tbd'].index)
    logger.info(f"Total number of games: {games.shape[0]}")
    logger.info(f"Games with user score as tbd: {games.shape[0] - tbd_cleaned_games.shape[0]}")
    logger.info(f"Games without user score as tbd: {tbd_cleaned_games.shape[0]}")

    cleaned_games = tbd_cleaned_games[tbd_cleaned_games["summary"].notna()]
    logger.info(f"All Games with summary as NaN: {tbd_cleaned_games.shape[0]}")
    logger.info(f"Only Games with summary as NaN: {tbd_cleaned_games.shape[0] - cleaned_games.shape[0]}")
    logger.info(f"Games with summary cleaned: {cleaned_games.shape[0]}")

    high_scored_games = cleaned_games[cleaned_games["metascore"] >= 97].filter(['title', 'metascore'])
    return high_scored_games.to_json(orient='records')


def linear_regression() -> float:
    logger = logging.getLogger(__name__)
    logger.info("Training a linear regression model")
    data = {'x': [1, 2, 3, 4, 5],
            'y': [2, 4, 6, 8, 10]}

    df = pd.DataFrame(data)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df[['x']], df['y'], test_size=0.2, random_state=42)

    # Train a linear regression model on the training set
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Use the trained model to make predictions on the testing set
    y_pred = lr.predict(X_test)

    # Evaluate the performance of the model
    mse = mean_squared_error(y_test, y_pred)
    logger.info(f"Mean Squared Error: {mse}")
    return mse


def logistic_regression() -> float:
    logger = logging.getLogger(__name__)
    logger.info("Training a logistic regression model")
    data = {'X1': [2, 3, 3, 4, 5],
            'X2': [3, 2, 4, 3, 5],
            'Y': [0, 0, 0, 1, 1]}

    df = pd.DataFrame(data)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df[['X1', 'X2']], df['Y'], test_size=0.2,
                                                        random_state=42)

    # Train a logistic regression model on the training set
    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    # Use the trained model to make predictions on the testing set
    y_pred = lr.predict(X_test)

    # Evaluate the performance of the model
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Accuracy: {accuracy}")
    return accuracy


def knn() -> float:
    # Create a small dataset for KNN regression
    logger = logging.getLogger(__name__)
    data = {'x': [1, 2, 3, 4, 5],
            'y': [2, 4, 6, 8, 10]}

    df = pd.DataFrame(data)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df[['x']], df['y'], test_size=0.2, random_state=42)

    # Train a KNN regression model on the training set
    k = 3  # number of neighbors to consider
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Use the trained model to make predictions on the testing set
    y_pred = knn.predict(X_test)

    # Evaluate the performance of the model
    mse = mean_squared_error(y_test, y_pred)
    logger.info(f"mse: {mse}")
    return mse


def decision_tree() -> float:
    logger = logging.getLogger(__name__)
    logger.info("decision tree")
    data = {'age': [22, 25, 47, 52, 21, 19, 62, 46, 36, 57],
            'gender': ['M', 'F', 'F', 'M', 'M', 'F', 'M', 'F', 'M', 'M'],
            'income': [25, 32, 55, 72, 15, 20, 80, 65, 52, 90],
            'is_student': [True, True, False, False, True, True, False, True, False, False],
            'bought_insurance': [0, 0, 1, 1, 0, 0, 1, 1, 1, 1]}

    df = pd.DataFrame(data)

    # Convert categorical features into numerical features
    df['gender'] = df['gender'].map({'M': 0, 'F': 1})
    df['is_student'] = df['is_student'].astype(int)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df.drop('bought_insurance', axis=1), df['bought_insurance'],
                                                        test_size=0.2, random_state=42)
    # Train a Decision Tree classification model on the training set
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)

    # Use the trained model to make predictions on the testing set
    y_pred = dt.predict(X_test)

    # Evaluate the performance of the model
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Accuracy: {accuracy}")
    return accuracy


def random_forest() -> float:
    logger = logging.getLogger(__name__)
    logger.info("random forest")
    # Create a small dataset for Random Forest classification with 5 variables
    data = {'age': [22, 25, 47, 52, 21, 19, 62, 46, 36, 57],
            'income': [25000, 30000, 50000, 70000, 20000, 15000, 90000, 60000, 40000, 80000],
            'gender': ['F', 'M', 'F', 'M', 'F', 'F', 'M', 'M', 'F', 'M'],
            'marital_status': ['Single', 'Single', 'Married', 'Married', 'Single', 'Single', 'Married', 'Married',
                               'Single', 'Married'],
            'bought_insurance': [0, 0, 1, 1, 0, 0, 1, 1, 1, 1]}

    df = pd.DataFrame(data)

    # Convert categorical variables into dummy variables
    df = pd.get_dummies(df, columns=['gender', 'marital_status'])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df.drop('bought_insurance', axis=1), df['bought_insurance'],
                                                        test_size=0.2, random_state=42)

    # Train a Random Forest classification model on the training set
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Use the trained model to make predictions on the testing set
    y_pred = rf.predict(X_test)

    # Evaluate the performance of the model
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Accuracy: {accuracy}")
    return accuracy


def naive_bayes() -> int:
    logger = logging.getLogger(__name__)
    logger.info("naive_bayes")
    data = {'Feature1': [0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
            'Feature2': [1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
            'Feature3': [0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
            'Target': ['No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No']}

    df = pd.DataFrame(data)

    # Split data into features and target variable
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create Naive Bayes classifier
    nb_classifier = GaussianNB()

    # Train the classifier on the training data
    nb_classifier.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = nb_classifier.predict(X_test)

    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Accuracy: {accuracy}")
    return accuracy


def k_means_clustering() -> int:
    logger = logging.getLogger(__name__)
    logger.info("k_means_clustering")
    return 0


def support_vector_machine() -> int:
    logger = logging.getLogger(__name__)
    logger.info("support_vector_machine")
    return 0


def principal_component_analysis() -> int:
    logger = logging.getLogger(__name__)
    logger.info("principal_component_analysis")
    return 0


def recommender_system() -> int:
    logger = logging.getLogger(__name__)
    logger.info("recommender_system")
    return 0


def natural_language_processing() -> int:
    logger = logging.getLogger(__name__)
    logger.info("natural_language_processing")
    return 0


def deep_learning() -> int:
    logger = logging.getLogger(__name__)
    logger.info("deep_learning")
    return 0

