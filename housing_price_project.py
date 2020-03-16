import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

from sklearn.model_selection import train_test_split, cross_val_score, \
    GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

housing = pd.read_csv('C:\\Users\\shu.zhan\\Downloads\\Ryerson_Project'
                      '\\house-prices-advanced-regression-techniques\\housing'
                      '.csv')
housing = housing.iloc[:, 3:]
print(housing.info())


def transform_rate(rate):
    if rate == 1:
        return ('Very Poor')
    elif rate == 2:
        return ('Poor')
    elif rate == 3:
        return ('Fair')
    elif rate == 4:
        return ('Below Average')
    elif rate == 5:
        return ('Average')
    elif rate == 6:
        return ('Above Average')
    elif rate == 7:
        return ('Good')
    elif rate == 8:
        return ('Very Good')
    elif rate == 9:
        return ('Excellent')
    elif rate == 10:
        return ('Very Excellent')


def transform_subclass(subclass):
    if subclass == 20:
        return ('1-STORY 1946 & NEWER ALL STYLES')
    elif subclass == 30:
        return ('1-STORY 1945 & OLDER')
    elif subclass == 40:
        return ('1-STORY W/FINISHED ATTIC ALL AGES')
    elif subclass == 45:
        return ('1-1/2 STORY - UNFINISHED ALL AGES')
    elif subclass == 50:
        return ('1-1/2 STORY FINISHED ALL AGES')
    elif subclass == 60:
        return ('2-STORY 1946 & NEWER')
    elif subclass == 70:
        return ('2-STORY 1945 & OLDER')
    elif subclass == 75:
        return ('2-1/2 STORY ALL AGES')
    elif subclass == 80:
        return ('SPLIT OR MULTI-LEVEL')
    elif subclass == 85:
        return ('SPLIT FOYER')
    elif subclass == 90:
        return ('DUPLEX - ALL STYLES AND AGES')
    elif subclass == 120:
        return ('1-STORY PUD (Planned Unit Development) - 1946 & NEWER')
    elif subclass == 150:
        return ('1-1/2 STORY PUD - ALL AGES')
    elif subclass == 160:
        return ('2-STORY PUD - 1946 & NEWER')
    elif subclass == 180:
        return ('PUD - MULTILEVEL - INCL SPLIT LEV/FOYER')
    elif subclass == 190:
        return ('2 FAMILY CONVERSION - ALL STYLES AND AGES')


housing['Overall Qual'] = housing['Overall Qual'].apply(transform_rate)
housing['Overall Cond'] = housing['Overall Cond'].apply(transform_rate)
housing['MS SubClass'] = housing['MS SubClass'].apply(transform_subclass)

housing = housing.drop(['Alley', 'Pool QC', 'Fence', 'Misc Feature', 'Fireplace Qu'], axis=1)

# Change categorical variables from object type to category type
for column in housing.select_dtypes(['object']).columns:
    housing[column] = housing[column].astype('category')

print(housing.info())

columns = list(housing.iloc[:, :-1].columns)
for column in columns:
    if str(housing[column].dtypes) != 'category':
        housing.plot(kind='scatter', x=column, y='SalePrice')
        print(column)
        plt.show()

housing.SalePrice.plot(kind='box')
plt.title('Distribution of SalePrice')
plt.show()

housing.SalePrice.plot(kind='hist',bins=50)
plt.title('Distribution of SalePrice')
plt.xlabel('SalePrice')
plt.show()

# logarithm transformation
housing.SalePrice=np.log(housing.SalePrice)
housing.SalePrice.plot(kind='hist',bins=50)
plt.title('Distribution of SalePrice')
plt.xlabel('SalePrice')
plt.show()



housing = housing.sort_values(by=['Yr Sold', 'Mo Sold'])
housing.reset_index(drop=True, inplace=True)
# housing.to_csv('C:\\Users\\shu.zhan\\Downloads\\Ryerson_Project\\house-prices'
#                '-advanced-regression-techniques\\housing_clean.csv')



# 6 training and test splits
housing['YearMonth'] = housing['Yr Sold'] * 100 + housing['Mo Sold']
print(housing.head())


def training_test_split(start_date, end_date):
    train = housing[housing['YearMonth'] < start_date].drop(['YearMonth'],
                                                            axis=1)
    test = housing[(housing['YearMonth'] >= start_date) & (housing[
                                                               'YearMonth'] < end_date)].drop(
        ['YearMonth'], axis=1)
    # These NA's indicate that the house just doesn't have it
    empty_means_without = ['Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure',
                           'BsmtFin Type 1',
                           'BsmtFin Type 2',
                           'Garage Type', 'Garage Finish', 'Garage Qual',
                           'Garage '
                           'Cond']
    for feature in empty_means_without:
        train[feature].cat.add_categories(['None'], inplace=True)
        train[feature].fillna('None', inplace=True)
    train['Mas Vnr Type'].fillna('None', inplace=True)
    train['Mas Vnr Area'].fillna(0, inplace=True)
    # Replace NA's in numeric variables with the mean
    train['Lot Frontage'].fillna(train['Lot Frontage'].mean(), inplace=True)
    train['Garage Yr Blt'][train['Garage Yr Blt'] == 2207] = 2007
    train['Garage Yr Blt'][train['Garage Cars'] == 0].fillna(2999, inplace=True)

    train.dropna(inplace=True)  # Drop any remaining NA's

    print(train.info())

    return (train, test)


train_1, test_1 = training_test_split(200804, 200904)
train_2, test_2 = training_test_split(200807, 200907)
train_3, test_3 = training_test_split(200810, 200910)
train_4, test_4 = training_test_split(200901, 201001)
train_5, test_5 = training_test_split(200904, 201004)
train_6, test_6 = training_test_split(200907, 201007)



def pearson_corr_fs(data):
    # Using Pearson Correlation

    cor = data.corr()
    # Correlation with output variable
    cor_target = abs(cor["SalePrice"])
    # Selecting highly correlated features
    relevant_features = cor_target[cor_target > 0.5]
    print(relevant_features)
    # print(relevant_features.index)
    # # features highly correlated with target variable
    df_corr = dataset[relevant_features.index]
    # Compute the correlation matrix

    corr = df_corr.corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(12, 10))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, annot=True,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.show()
    return relevant_features.index


def linear_regression(data, t):
    X = data.drop('SalePrice', axis=1)
    y = data.SalePrice
    X_test = t.drop('SalePrice', axis=1)
    y_test = t.SalePrice
    linear_reg = LinearRegression()

    MSEs = cross_val_score(linear_reg, X, y,
                           scoring='neg_mean_squared_error', cv=10)
    MSE = np.median(MSEs)
    print('Mean Squared Error:', MSE)
    linear_reg.fit(X, y)
    y_pred = linear_reg.predict(X_test)
    lin_r = linear_reg.score(X_test, y_test)
    lin_mse = mean_squared_error(y_pred, y_test)
    return lin_mse, lin_r


def lasso_fs(data):
    # Lasso
    X = data.drop('SalePrice', axis=1)
    y = data.SalePrice
    reg = LassoCV()
    reg.fit(X, y)
    print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
    print("Best score using built-in LassoCV: %f" % reg.score(X, y))
    coef = pd.Series(reg.coef_, index=X.columns)

    print("Lasso picked " + str(
        sum(coef != 0)) + " variables and eliminated the other " + str(
        sum(coef == 0)) + " variables")
    imp_coef = coef[coef != 0].sort_values()
    print(imp_coef)
    plt.rcParams['figure.figsize'] = (8.0, 10.0)
    imp_coef.plot(kind="barh")
    plt.title("Feature importance using Lasso Model")
    plt.show()
    return imp_coef.index.values.tolist()


def lasso_regression(X, y, X_test, y_test):
    # X = data.drop('SalePrice', axis=1)
    # y = data.SalePrice
    lasso = Lasso()
    parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
    lasso_reg = GridSearchCV(lasso, parameters,
                             scoring='neg_mean_squared_error',
                             cv=10)
    lasso_reg.fit(X, y)

    print(lasso_reg.best_params_)
    print(lasso_reg.best_score_)

    lasso = Lasso(alpha=lasso_reg.best_params_['alpha'])
    MSEs = cross_val_score(lasso, X, y,
                           scoring='neg_mean_squared_error',
                           cv=10)
    MSE = np.median(MSEs)
    print('Mean Squared Error:', MSE)
    lasso.fit(X, y)
    y_pred = lasso.predict(X_test)
    lasso_r = lasso.score(X_test, y_test)
    lasso_mse = mean_squared_error(y_pred, y_test)
    return lasso_mse, lasso_r


def randomforest_fs(data):
    X = data.drop('SalePrice', axis=1)
    y = data.SalePrice
    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators=1000)
    # Train the model on training data
    rf.fit(X, y)
    # Get numerical feature importances
    feature_list = X.columns
    importances = list(rf.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for
                           feature, importance in
                           zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key=lambda x: x[1],
                                 reverse=True)
    selected_features = []
    # Print out the feature and importances
    for pair in feature_importances:
        if pair[1] > 0.01:
            print('Variable: {:25} Importance: {}'.format(
                *pair))
            selected_features.append(pair[0])
    return selected_features


def randomforest_regression(X, y, X_test, y_test):
    # X = data.drop('SalePrice', axis=1)
    # y = data.SalePrice
    rf = RandomForestRegressor()
    parameters = {'n_estimators': [100,200,300,500,1000]}
    rf_reg = GridSearchCV(rf, parameters,
                          scoring='neg_mean_squared_error',
                          cv=10)
    rf_reg.fit(X, y)

    print(rf_reg.best_params_)
    print(rf_reg.best_score_)

    rf = RandomForestRegressor(n_estimators=rf_reg.best_params_['n_estimators'])
    MSEs = cross_val_score(rf, X, y,
                           scoring='neg_mean_squared_error',
                           cv=10)
    MSE = np.median(MSEs)
    print('Mean Squared Error:', MSE)
    rf.fit(X, y)
    y_pred = rf.predict(X_test)
    rf_r = rf.score(X_test, y_test)
    rf_mse = mean_squared_error(y_pred, y_test)
    return rf_mse, rf_r


MSE_result = pd.DataFrame(columns = ['mse','r'])

dataset = pd.get_dummies(train_6) # repete for training set 1 to 6
test = pd.get_dummies(test_6)     # repete for testing set 1 to 6
print(dataset.info())
# X = dataset.drop('SalePrice', axis=1)  # independent columns
# y = dataset['SalePrice']  # target column


# linear regression
# # drop 'TotRms AbvGrd' and 'Garage Cars'
df = dataset[pearson_corr_fs(dataset)]
print(df.columns)
df_linear = df.drop(['Garage Cars', '1st Flr SF','Garage Yr '
                                                                  'Blt'
                                                                  ''],
                    axis=1)
test_linear = test[list(df_linear.columns)]
test_linear.dropna(inplace=True)
test_linear = test_linear.sample(n=550)
MSE_result.loc['linear_regression','mse'],MSE_result.loc['linear_regression','r'
] = linear_regression(df_linear,
                                                                   test_linear)



# lasso regression

df_lasso = dataset[lasso_fs(dataset)]
test_columns = list(df_lasso.columns)
test_columns.append('SalePrice')
print(test_columns)
test_lasso = test[test_columns]
test_lasso.dropna(inplace=True)
test_lasso = test_lasso.sample(n=550)
print(test_lasso.info())
MSE_result.loc['lasso_regression','mse'],MSE_result.loc['lasso_regression',
                                                        'r'] = \
    lasso_regression(df_lasso,
                                                                dataset.SalePrice,test_lasso.drop('SalePrice', axis=1),test_lasso['SalePrice'])


# RandomForest

df_random_forest = dataset[randomforest_fs(dataset)]
test_columns = list(df_random_forest.columns)
test_columns.append('SalePrice')
test_random_forest = test[test_columns]
test_random_forest.dropna(inplace=True)
test_random_forest = test_random_forest.sample(n=550)
MSE_result.loc['random_forest_regression','mse'],MSE_result.loc[
    'random_forest_regression','r'] = \
    randomforest_regression(df_random_forest, dataset.SalePrice,
                            test_random_forest.drop('SalePrice', axis=1),
                            test_random_forest['SalePrice'])

print(MSE_result)

# # OLS
#
# X_train, X_test, y_train, y_test = train_test_split(df.drop('SalePrice', axis=1), df.SalePrice, test_size=0.3)
# #Adding constant column of ones, mandatory for sm.OLS model
# X_train = sm.add_constant(X_train)
# #Fitting sm.OLS model
# model = sm.OLS(y_train,X_train).fit()
# print(model.summary())
#
# X_test = sm.add_constant(X_test)
# y_pred = model.predict(X_test)
# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))



