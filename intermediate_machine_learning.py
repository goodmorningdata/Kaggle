import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

def introduction():
    # Read the data
    X_full = pd.read_csv('train.csv', index_col='Id')
    X_test_full = pd.read_csv('test.csv', index_col='Id')

    # Obtain target and predictors
    y = X_full.SalePrice
    features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath',     'BedroomAbvGr', 'TotRmsAbvGrd']
    X = X_full[features].copy()
    X_test = X_test_full[features].copy()

    # Split training and test data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

    # Define 5 different models
    model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
    model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
    model_3 = RandomForestRegressor(n_estimators=100, criterion='mae',
                                    random_state=0)
    model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20,
                                    random_state=0)
    model_5 = RandomForestRegressor(n_estimators=100, max_depth=7,
                                    random_state=0)

    def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
        model.fit(X_t, y_t)
        preds = model.predict(X_v)
        return mean_absolute_error(y_v, preds)

    models = [model_1, model_2, model_3, model_4, model_5]
    for i in range(0, len(models)):
        mae = score_model(models[i])
        print('Model %d MAE: %d' % (i+1, mae))

def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

def dealing_with_missing_values():
    melbourne_data = pd.read_csv('melb_data.csv')

    y = melbourne_data.Price
    X = melbourne_data.select_dtypes(exclude=['object'])
    X = X.drop(['Price'], axis=1)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

    # ** Approach 1 - Drop cols with missing values.

    # Get names of columns with missing values.
    cols_with_missing = [col for col in X_train.columns
                         if X_train[col].isnull().any()]

    # Drop missing columns
    reduced_X_train = X_train.drop(cols_with_missing, axis=1)
    reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

    print('MAE from approach 1 (Drop columns with missing values):')
    print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))

    # ** Approach 2 - Imputation
    my_imputer = SimpleImputer()
    imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
    imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

    # Imputation removed column names; put them back
    imputed_X_train.columns = X_train.columns
    imputed_X_valid.columns = X_valid.columns

    print('MAE from approach 2 (Imputation):')
    print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))

    # ** Approach 3 - An extension to imputation
    X_train_plus = X_train.copy()
    X_valid_plus = X_valid.copy()

    # Make new columns indicating what will be imputed
    for col in cols_with_missing:
        X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
        X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

    # Imputation
    imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
    imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

    # Imputation removed column names; put them back
    imputed_X_train_plus.columns = X_train_plus.columns
    imputed_X_valid_plus.columns = X_valid_plus.columns

    print('MAE from approach 3 (Imputation extension):')
    print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))

    # Shape of training data (num_rows, num_columns)
    print(X_train.shape)

    # Number of missing values in each column of training data
    missing_val_count_by_column = (X_train.isnull().sum())
    print(missing_val_count_by_column[missing_val_count_by_column > 0])

def missing_values():
    # Read the data
    X_full = pd.read_csv('train.csv', index_col='Id')
    X_test_full = pd.read_csv('test.csv', index_col='Id')

    # Remove rows with missing target and separate target from predictors
    X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = X_full.SalePrice
    X_full.drop(['SalePrice'], axis=1, inplace=True)

    # Use only numerical predictors
    X = X_full.select_dtypes(exclude=['object'])
    X_test = X_test_full.select_dtypes(exclude=['object'])

    # Split training and test data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

    print(X_train.head())

    # Shape of the training data
    print(X_train.shape)

    # Number of missing values in each column of traiing data
    missing_val_count_by_column = (X_train.isnull().sum())
    print(missing_val_count_by_column[missing_val_count_by_column > 0])

    # * Approach 1 - drop columns with missing values
    cols_with_missing = [col for col in X_train.columns
                         if X_train[col].isnull().any()]
    reduced_X_train = X_train.drop(cols_with_missing, axis=1)
    reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

    print('MAE (Drop columns with missing values):')
    print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))

    # * Approach 2 - imputation
    my_imputer = SimpleImputer()
    imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
    imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))
    imputed_X_train.columns = X_train.columns
    imputed_X_valid.columns = X_valid.columns

    print('MAE (Imputation):')
    print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))

    # * Approach 3 - My choice
    # LotFrontage
    # MasVnrArea
    # GarageYrBlt
    # Replace LotFrontage with mean
    #final_X_train = X_train.copy()
    #final_X_valid = X_valid.copy()
    #final_X_train.drop(['MasVnrArea', 'GarageYrBlt'], axis=1)
    #final_X_valid.drop(['MasVnrArea', 'GarageYrBlt'], axis=1)
    final_X_train = X_train.drop(['MasVnrArea', 'GarageYrBlt', ], axis=1)
    final_X_valid = X_valid.drop(['MasVnrArea', 'GarageYrBlt', ], axis=1)
    cols = final_X_train.columns
    my_imputer = SimpleImputer()
    final_X_train = pd.DataFrame(my_imputer.fit_transform(final_X_train))
    final_X_valid = pd.DataFrame(my_imputer.transform(final_X_valid))
    final_X_train.columns = cols
    final_X_valid.columns = cols

    # imp = Imputer(missing_values='NaN', strategy='mean')
    # imp.fit(final_X_train['LotFrontage'])
    # q = imp.fit_transform(final_X_train['LotFrontage']).T
    # final_X_train['LotFrontage'] = q
    # print(final_X_train)

    print('MAE (My Choice):')
    print(score_dataset(final_X_train, final_X_valid, y_train, y_valid))

def dealing_with_categorical_data():
    # Read the data and separate target from predictors
    data = pd.read_csv('melb_data.csv')
    y = data.Price
    X = data.drop(['Price'], axis=1)

    # Divide data into training and validation subsets
    X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

    # Drop columns with missing values (simplest approach)
    cols_with_missing = [col for col in X_train_full.columns if X_train_full[col].isnull().any()]
    X_train_full.drop(cols_with_missing, axis=1, inplace=True)
    X_valid_full.drop(cols_with_missing, axis=1, inplace=True)

    # "Cardinality" means the number of unique values in a column
    # Select categorical columns with relatively low cardinality (convenient but arbitrary)
    low_cardinality_cols = [cname for cname in X_train_full.columns
        if X_train_full[cname].nunique() < 10 and
           X_train_full[cname].dtype == "object"]

    # Select numerical columns
    numerical_cols = [cname for cname in X_train_full.columns
        if X_train_full[cname].dtype in ['int64', 'float64']]

    # Keep selected columns only
    my_cols = low_cardinality_cols + numerical_cols
    X_train = X_train_full[my_cols].copy()
    X_valid = X_valid_full[my_cols].copy()

    # Get list of catagorical variables
    s = (X_train.dtypes == 'object')
    object_cols = list(s[s].index)
    print('Categorical variables:')
    print(object_cols)

    # ** Approach 1 - drop categorical variables
    drop_X_train = X_train.select_dtypes(exclude=['object'])
    drop_X_valid = X_valid.select_dtypes(exclude=['object'])
    print('MAE from Approach 1 (drop categorical variables):')
    print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))

    # ** Approach 2 - label encoding
    label_X_train = X_train.copy()
    label_X_valid = X_valid.copy()
    # Apply label encoder to each column with categorical data.
    label_encoder = LabelEncoder()
    for col in object_cols:
        label_X_train[col] = label_encoder.fit_transform(X_train[col])
        label_X_valid[col] = label_encoder.transform(X_valid[col])
    print('MAE from Approach 2 (label encoding):')
    print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))

    # ** Approach 3 - one-hot encoding
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
    OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))
    # One-hot encoding removed index; put it back
    OH_cols_train.index = X_train.index
    OH_cols_valid.index = X_valid.index
    # Remove categorical columns and replace with one-hot encoding
    num_X_train = X_train.drop(object_cols, axis=1)
    num_X_valid = X_valid.drop(object_cols, axis=1)
    OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
    OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)
    print('MAE from Approach 3 (one-hot encoding):')
    print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))

def categorical_variables():
    # Read the data, remove rows with missing target, and separate target and predictors.
    X = pd.read_csv('train.csv', index_col='Id')
    X_test = pd.read_csv('test.csv', index_col='Id')
    X.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = X.SalePrice
    X.drop(['SalePrice'], axis=1, inplace=True)

    # Drop columns with missing values
    cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
    X.drop(cols_with_missing, axis=1, inplace=True)
    X_test.drop(cols_with_missing, axis=1, inplace=True)

    # Split training and test data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

    # Drop columns with categorical data
    drop_X_train = X_train.select_dtypes(exclude=['object'])
    drop_X_valid = X_valid.select_dtypes(exclude=['object'])
    print('MAE from Approach 1 (drop categorical variables):')
    print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))

    # Drop categorical variables that don't contain the same set of values in the training and test data.
    object_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']
    good_label_cols = [col for col in object_cols
        if set(X_train[col]) == set(X_valid[col])]
    bad_label_cols = list(set(object_cols) - set(good_label_cols))
    print('Categorical columns that will be label encoded:', good_label_cols)
    print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)
    label_X_train = X_train.drop(bad_label_cols, axis=1)
    label_X_valid = X_valid.drop(bad_label_cols, axis=1)

    # Apply label encoder
    label_encoder = LabelEncoder()
    for col in good_label_cols:
        label_X_train[col] = label_encoder.fit_transform(label_X_train[col])
        label_X_valid[col] = label_encoder.transform(label_X_valid[col])
    print("MAE from Approach 2 (Label Encoding):")
    print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))

    # Get number of unique entries in each column with categorical data
    object_nunique = list(map(lambda col: X_train[col].nunique, object_cols))
    d = dict(zip(object_cols, object_nunique))
    # Print number of unique entries by column, in ascending order
    #print(sorted(d.items(), key=lambda x: x[1]))

    low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]
    high_cardinality_cols = list(set(object_cols) - set(low_cardinality_cols))
    print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)
    print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols)

    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))
    OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))
    OH_cols_train.index = X_train.index
    OH_cols_valid.index = X_valid.index
    num_X_train = X_train.drop(object_cols, axis=1)
    num_X_valid = X_valid.drop(object_cols, axis=1)
    OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
    OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)
    print("MAE from Approach 3 (One-Hot Encoding):")
    print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))

def pipelines():
    # A way to keep data preprocessing and modelling data organized.
    # Construct in three steps:
    # 1) Define preprocessing steps.
    # 2) Define the model.
    # 3) Create and evaluate the pipeline.
    data = pd.read_csv('melb_data.csv')
    y = data.Price
    X = data.drop(['Price'], axis=1)
    X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
    # Select columns with low cardinality
    categorical_cols = [col for col in X.columns
        if X[col].nunique() < 10 and X[col].dtype == 'object']
    # Select numeric columns
    numerical_cols = [col for col in X.columns
        if X[col].dtype in ['int64', 'float64']]
    # Keep selected columns only
    my_cols = categorical_cols + numerical_cols
    X_train = X_train_full[my_cols].copy()
    X_valid = X_valid_full[my_cols].copy()

    # Step 1 - Define preprocessing steps
    # Preprocessing for categorical data
    numerical_transformer = SimpleImputer(strategy='constant')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    # Step 2 - Define the model
    model = RandomForestRegressor(n_estimators=100, random_state=0)

    # Step 3 - Create and evaluate the pipeline
    # Bundle preprocessing and modeling code in a pipeline
    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('model', model)])
    # Preprocessing of training data, fit model
    my_pipeline.fit(X_train, y_train)
    # Preprocessing of validation data, get predictions
    preds = my_pipeline.predict(X_valid)
    # Evaluate the model
    score = mean_absolute_error(y_valid, preds)
    print('MAE: ', score)

def pipelines_exercise():
    # Read the data and split training and test data.
    X_full = pd.read_csv('train.csv', index_col='Id')
    X_test_full = pd.read_csv('test.csv', index_col='Id')
    X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = X_full.SalePrice
    X_full.drop(['SalePrice'], axis=1, inplace=True)
    X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, train_size=0.8, test_size=0.2, random_state=0)

    # Select categorical columns with low cardinality
    categorical_cols = [col for col in X_train_full.columns
        if X_train_full[col].nunique() < 10 and
        X_train_full[col].dtype == 'object']
    # Select numerical columns
    numerical_cols = [col for col in X_train_full.columns
        if X_train_full[col].dtype in ['int64', 'float64']]
    my_cols = categorical_cols + numerical_cols
    X_train = X_train_full[my_cols].copy()
    X_valid = X_valid_full[my_cols].copy()
    X_test = X_test_full[my_cols].copy()

    # Preprocessing
    #numerical_transformer = SimpleImputer(strategy='constant', fill)
    numerical_transformer = SimpleImputer(strategy='mean')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)])

    # Define model
    model = RandomForestRegressor(n_estimators=150, random_state=0)

    # Bundle preprocessing and modeling code in a pipeline
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', model)])

    # Preprocessing of training data, fit model
    clf.fit(X_train, y_train)
    # Preprocessing of validation data, get predictions
    preds = clf.predict(X_valid)

    print('MAE: ', mean_absolute_error(y_valid, preds))

    preds_test = clf.predict(X_test)

def cross_validation():
    # Run our modeling process on different subsets of the data to get multiple measures of model quality. Break the data into equally sized "folds". Use every fold once as the validation set. Takes longer to run. Use for small datasets. For large datasets use only a single validation set.

   # Read the data
    data = pd.read_csv('melb_data.csv')
    cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
    X = data[cols_to_use]
    y = data.Price
    my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(n_estimators=50, random_state=0))])

    # Multiply by -1 since sklearn calculates negative MAE
    scores = -1 * cross_val_score(my_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
    print('MAE scores:\n', scores)
    print('Average MAE score (across experiments):')
    print(scores.mean())

def get_score(n_estimators):
    my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(n_estimators=n_estimators, random_state=0))])
    scores = -1 * cross_val_score(my_pipeline, X, y, cv=3, scoring='neg_mean_absolute_error')
    return scores.mean()

def cross_validation_exercise():
    # Read the data
    train_data = pd.read_csv('train.csv', index_col='Id')
    test_data = pd.read_csv('test.csv', index_col='Id')
    train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y=train_data.SalePrice
    train_data.drop(['SalePrice'], axis=1, inplace=True)
    # Select numeric columns only
    numeric_cols = [col for col in train_data.columns if train_data[col].dtype in ['int64', 'float64']]
    X = train_data[numeric_cols].copy()
    X_test = test_data[numeric_cols].copy()

    my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(n_estimators=50, random_state=0))])

    scores = -1 * cross_val_score(my_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
    print('Average MAE score (across experiments):')
    print(scores.mean())

    def get_score(n_estimators):
        my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
            ('model', RandomForestRegressor(n_estimators=n_estimators, random_state=0))])
        scores = -1 * cross_val_score(my_pipeline, X, y, cv=3, scoring='neg_mean_absolute_error')
        return scores.mean()

    results = {}
    for i in range(50, 450, 50):
        results[i] = get_score(i)

    print(results)

def gradient_boosting_example():
    # Random Forest is an "Ensemble" method = combine the predictions of several models.
    # Gradient Boosting - a method that goes through cycles to iteratively add models into an ensemble. Cycle = 1. naive model, 2. make predictions, 3. calculate loss function, 4. use loss function to fit a new model to add to ensemble. determine model parameters to reduce loss. 5. add new model to ensemble.
    data = pd.read_csv('melb_data.csv')
    cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
    X = data[cols_to_use]
    y = data.Price
    X_train, X_valid, y_train, y_valid = train_test_split(X,y)

    # XGBoost = extreme gradient boosting. Implementation of gradient boosting with several additional features focused on performance and speed.
    my_model = XGBRegressor()
    my_model.fit(X_train, y_train)
    predictions = my_model.predict(X_valid)
    print('Mean Absolute Error: ' + str(mean_absolute_error(predictions, y_valid)))

    # **** Parameter tuning ****
    # n_estimators = how many times to go through the modeling cycle described above. Too low causes underfitting, too high causes overfitting. Typical 100-1000.
    my_model = XGBRegressor(n_estimators=500)
    my_model.fit(X_train, y_train)

    # early_stopping_rounds = automatically find the ideal value for n_estimators. Set high value for n_estimators and use early_stopping_rounds. How many rounds of straight deterioration to allow before stopping. 5 is a reasonable choice. eval_set parameter sets aside some data for calculating the validation score to check if score has stopped improving.
    my_model = XGBRegressor(n_estimators=500)
    my_model.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_valid, y_valid)], verbose=False)

    # learning_rate = multiply the predictions from each model by a small number before adding them in. Each tree we add to the ensemble helps us less. Can set a higher value for n_estimators without overfitting. In general a small learning rate and a large number of estimators will yield more accurate models. Default is learning_rate=0.1
    my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
    my_model.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_valid, y_valid)], verbose=False)

    # n_jobs = Use parallelism to build models faster. Does not help small datasets. Does not improve model.
    my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
    my_model.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_valid, y_valid)], verbose=False)

def xgboost_exercise():
    X = pd.read_csv('train.csv', index_col='Id')
    X_test_full = pd.read_csv('test.csv', index_col='Id')
    X.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = X.SalePrice
    X.drop(['SalePrice'], axis=1, inplace=True)
    X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

    # Select categorical columns with low cardinaltiy
    low_cardinality_cols = [col for col in X_train_full.columns if X_train_full[col].nunique()<10 and X_train_full[col].dtype == 'object']
    # Select numeric columns
    numeric_cols = [col for col in X_train_full.columns if X_train_full[col].dtype in ['int64', 'float64']]
    my_cols = low_cardinality_cols + numeric_cols
    X_train = X_train_full[my_cols].copy()
    X_valid = X_valid_full[my_cols].copy()
    X_test = X_test_full[my_cols].copy()

    # One-hot encode the data (to shorten the code we use pandas)
    X_train = pd.get_dummies(X_train)
    X_valid = pd.get_dummies(X_valid)
    X_test = pd.get_dummies(X_test)
    X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
    X_train, X_test = X_train.align(X_test, join='left', axis=1)

    # Build and the fit the model
    my_model_1 = XGBRegressor(random_state=0)
    my_model_1.fit(X_train, y_train)
    predictions_1 = my_model_1.predict(X_valid)
    mae_1 = mean_absolute_error(predictions_1, y_valid)
    print("Mean Absolute Error 1:" , mae_1)

    my_model_2 = XGBRegressor(n_estimators=500, random_state=0)
    #my_model_2 = XGBRegressor(n_estimators=450, random_state=0)
    my_model_2.fit(X_train, y_train)
    #my_model_2.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_valid, y_valid)], verbose=False)
    predictions_2 = my_model_2.predict(X_valid)
    mae_2 = mean_absolute_error(predictions_2, y_valid)
    print("Mean Absolute Error 2:" , mae_2)

    my_model_3 = XGBRegressor(n_estimators=5, random_state=0)
    my_model_3.fit(X_train, y_train)
    predictions_3 = my_model_3.predict(X_valid)
    mae_3 = mean_absolute_error(predictions_3, y_valid)
    print("Mean Absolute Error 3:" , mae_3)

def data_leakage_example():
    # Data leakage occurs when you training data contains information about the target, but similar data will not be available when the model is used for prediction. Causes a model to look accurate until you use it on real data.
    # Target Leakage = occurs when your predictors include data that will not be available at the time you make predictions. Think about target leakage in terms of the timing or chronological order that data becomes available. To prevent this type of data leakage, any variable updated (or created) after the target value is realized should be excluded.
    # Train-Test Contamination = aren't careful to distinguish training data from validation data. Don't let the validation data affect the preprocessing behavior. Can't generalize to new data.

#introduction()
#dealing_with_missing_values()
#missing_values()
#dealing_with_categorical_data()
#categorical_variables()
#pipelines()
#pipelines_exercise()
#cross_validation()
#cross_validation_exercise()
#gradient_boosting_example()
#xgboost_exercise()
data_leakage_example()

'''
Three approaches to deal with missing values:
1. A simple option: Drop columns with missing values.
   Simple, but you can lose important data. A column with just one missing value will be dropped.

2. A better option: Imputation.
   Fills in the missing values with some number. i.e. The mean value.

3. An extension to imputation.
   Impute the missing values, and then add a column that shows the location of the imputed entries. i.e. "Bed_was_missing" column with T/F values.
'''
'''
Three approaches to deal with categorical data:
1. Drop categorical variables.
   Simple but you lose useful information.
2. Label encoding.
   Assigns each unique value to a different integer. Good for ordinal variables - those with a clear ordering.
3. One-hot encoding.
   Creates new columns indicating the presence or absence of each possible value in the original data. Does not assume ordering of the categories - nominal variables.
   - Set handle_unknown='ignore' to avoid errors when the validation data contains classes that aren't represented in the training data,
   - Set sparse=False ensures that the encoded columns are returned as a numpy array (instead of a sparse matrix).
In general, one-hot encoding (Approach 3) will typically perform best, and dropping the categorical columns (Approach 1) typically performs worst, but it varies on a case-by-case basis.

We refer to the number of unique entries of a categorical variable as the cardinality of that categorical variable.  For instance, the 'Street' variable has cardinality 2.
'''
