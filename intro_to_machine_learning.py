import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def intro():
    filename = 'melb_data.csv'
    melbourne_data = pd.read_csv(filename)
    print(melbourne_data.describe())

def explore_your_data():
    iowa_file_path = 'train.csv'
    home_data = pd.read_csv(iowa_file_path)
    print(home_data.describe())

    print('What is the average lot size (rounded to nearest integer)?')
    print(int(round(home_data.LotArea.mean(),0)))

    print('As of today, how old is the newest home (current year - the date in which it was built)')
    home_data['Age'] = 2019 - home_data.YearBuilt
    print(home_data.Age.min())

def first_machine_learning_model():
    filename = 'melb_data.csv'
    melbourne_data = pd.read_csv(filename)

    print(melbourne_data.columns)

    melbourne_data = melbourne_data.dropna(axis=0)

    # Selecting the prediction target
    y = melbourne_data.Price

    # Choosing features
    melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
    X = melbourne_data[melbourne_features]

    # Define and fit model
    melbourne_model = DecisionTreeRegressor(random_state=1)
    melbourne_model.fit(X,y)

    print('Making predictions for the following 5 houses:')
    print(X.head())
    print('The predictions are:')
    print(melbourne_model.predict(X.head()))

def model_building_exercise():
    iowa_file_path = 'train.csv'
    home_data = pd.read_csv(iowa_file_path)

    # Specify prediction target and create X
    print(home_data.columns)
    y = home_data.SalePrice
    features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
    X = home_data[features]

    # Specify and fit the model
    iowa_model = DecisionTreeRegressor(random_state=1)
    iowa_model.fit(X,y)

    # Make predictions
    predictions = iowa_model.predict(X)
    print(predictions)

    # Compare to actual
    print('Making predictions for the following 5 houses:')
    print(X.head())
    print('The predictions are:')
    print(iowa_model.predict(X.head()))

def model_validation():
    iowa_file_path = 'train.csv'
    home_data = pd.read_csv(iowa_file_path)
    y = home_data.SalePrice
    feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
    X = home_data[feature_columns]

    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

    # Specify and fit the model
    iowa_model = DecisionTreeRegressor(random_state=1)
    iowa_model.fit(train_X, train_y)

    val_predictions = iowa_model.predict(val_X)

    # print the top few validation predictions
    print(val_predictions)
    # print the top few actual prices from validation data
    print(val_y)

    val_mae = mean_absolute_error(val_y, val_predictions)
    print(val_mae)

    # Try different models
    for max_leaf_nodes in [5, 50, 500, 5000]:
        my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
        print("Max leaf nodes: %d \t\t Mean Absolute Error: %d" %(max_leaf_nodes, my_mae))

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae

def underfitting_overfitting():
    iowa_file_path = 'train.csv'
    home_data = pd.read_csv(iowa_file_path)
    y = home_data.SalePrice
    features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
    X = home_data[features]

    # Split into validation and training data
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

    # Specify and Fit Model
    iowa_model = DecisionTreeRegressor(random_state=1)
    iowa_model.fit(train_X, train_y)

    # Make validation predictions and calculate mean absolute error
    val_predictions = iowa_model.predict(val_X)
    val_mae = mean_absolute_error(val_predictions, val_y)
    print("Validation MAE: {:,.0f}".format(val_mae))

    candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
    mae_list = []
    for leaf_node in candidate_max_leaf_nodes:
        mae = get_mae(leaf_node, train_X, val_X, train_y, val_y)
        mae_list.append(mae)

    best_tree_size = candidate_max_leaf_nodes[mae_list.index(min(mae_list))]

    final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=0)
    final_model.fit(X, y)

def random_forest_model():
    iowa_file_path = 'train.csv'
    home_data = pd.read_csv(iowa_file_path)
    y = home_data.SalePrice

    features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
    X = home_data[features]

    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

    iowa_model = DecisionTreeRegressor(random_state=1)
    iowa_model.fit(train_X, train_y)

    # Make validation predictions and calculate mean absolute error
    val_predictions = iowa_model.predict(val_X)
    val_mae = mean_absolute_error(val_predictions, val_y)
    print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

    # Using best value for max_leaf_nodes
    iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
    iowa_model.fit(train_X, train_y)
    val_predictions = iowa_model.predict(val_X)
    val_mae = mean_absolute_error(val_predictions, val_y)
    print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))

    rf_model = RandomForestRegressor(random_state=1)
    rf_model.fit(train_X, train_y)
    rf_val_mae = mean_absolute_error(rf_model.predict(val_X), val_y)

def machine_learning_competition():
    iowa_file_path = 'train.csv'
    home_data = pd.read_csv(iowa_file_path)

    print (home_data.columns)
    print (home_data.dtypes)

    # Create X and y
    y = home_data.SalePrice
    features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'LotFrontage', 'OverallQual', 'OverallCond']

    X = home_data[features]
    X.fillna(value=0, inplace=True)

    import numpy as np
    print('** nan, inf, -inf values:')
    print(X[X.isin([np.nan, np.inf, -np.inf]).any(1)])

    # Split into validation and training data
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

    # Build a Random Forest model and train it on all of X and y.
    rf_model_on_full_data = RandomForestRegressor(random_state=1)
    rf_model_on_full_data.fit(X, y)

    test_data_path = 'test.csv'
    test_data = pd.read_csv(test_data_path)
    test_X = test_data[features]

    test_X.fillna(value=0, inplace=True)
    print('** nan, inf, -inf values:')
    print(test_X[test_X.isin([np.nan, np.inf, -np.inf]).any(1)])

    test_preds = rf_model_on_full_data.predict(test_X)

#intro()
#explore_your_data()
#first_machine_learning_model()
#model_building_exercise()
#model_validation()
#underfitting_overfitting()
#random_forest_model()
machine_learning_competition()
