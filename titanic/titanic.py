import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=50, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

def main():
    X = pd.read_csv('train.csv', index_col='PassengerId')
    X_test = pd.read_csv('test.csv', index_col='PassengerId')
    y = X.Survived
    X.drop(['Survived'], axis=1, inplace=True)

    # # Split training and test data.
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)

    # Which columns have missing values?
    cols_with_missing = [col for col in X.columns
                        if X[col].isnull().any()]
    print('Columns with missing values:')
    print(cols_with_missing)

    # Numeric columns are:
    # - Pclass (ticket class)
    # - Age
    # - SibSp (# of siblings and spouses aboard)
    # - Parch (# of parents and children aboard)
    # - Fare
    numeric_cols = [col for col in X.columns
                   if X[col].dtype in ['float64', 'int64']]
    print('Numeric columns:')
    print(numeric_cols)

    # Categorical columns
    categorical_cols = [col for col in X.columns
                       if X[col].dtype in ['object']]
    print('Categorical columns:')
    print(categorical_cols)

    # Use only numeric columns to start
    numeric_X_train = X_train[numeric_cols].copy()
    numeric_X_valid = X_valid[numeric_cols].copy()

    print(numeric_X_train)
    print(type(numeric_X_train))

    # Delete Age column.
    reduced_X_train = numeric_X_train.drop(['Age'], axis=1)
    reduced_X_valid = numeric_X_valid.drop(['Age'], axis=1)

    print(reduced_X_train)
    print(type(reduced_X_train))

    print('MAE (delete Age column):')
    print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))

    # Replace missing age values with mean age.
    my_imputer = SimpleImputer(strategy='mean')
    imputed_X_train = pd.DataFrame(my_imputer.fit_transform(numeric_X_train))
    imputed_X_valid = pd.DataFrame(my_imputer.transform(numeric_X_valid))
    imputed_X_train.columns = numeric_X_train.columns
    imputed_X_valid.columns = numeric_X_valid.columns

    print('MAE (impute Age column):')
    print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))

if __name__ == '__main__':
    main()
