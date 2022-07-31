import pickle
from math import sqrt

import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

data = pd.read_csv("Video_Games_Sales_as_at_22_Dec_2016.csv")
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)

data = data.rename(columns={"Year_of_Release": "Year", "NA_Sales": "NA", "EU_Sales": "EU", "JP_Sales": "JP",
                            "Other_Sales": "Other", "Global_Sales": "Global"})
data = data[data["Year"].notnull()]
data = data[data["Genre"].notnull()]
data["Year"] = data["Year"].apply(int)
data["Age"] = 2018 - data["Year"]

data["User_Score"] = data["User_Score"].replace("tbd", np.nan).astype(float)
data["Country"] = data[["NA", "EU", "JP", "Other"]].idxmax(1, skipna=True)


def rm_outlier(df, list_of_keys):
    df_out = df
    for key in list_of_keys:
        first_quartile = df_out[key].describe()['25%']
        third_quartile = df_out[key].describe()['75%']

        mid_quartile = third_quartile - first_quartile
        removed = df_out[(df_out[key] <= (first_quartile - 3 * mid_quartile)) |
                         (df_out[key] >= (third_quartile + 3 * mid_quartile))]

        df_out = df_out[(df_out[key] > (first_quartile - 3 * mid_quartile)) &
                        (df_out[key] < (third_quartile + 3 * mid_quartile))]

        return df_out, removed


data, rmvd_global = rm_outlier(data, ["Global"])

data['Has_Score'] = data['User_Score'].notnull() * data['Critic_Score'].notnull()
rmvd_global['Has_Score'] = rmvd_global['User_Score'].notnull() & rmvd_global['Critic_Score'].notnull()

scored = data.dropna(subset=["User_Score", "Critic_Score", "Rating"])
scored, rmvd_User_Count = rm_outlier(scored, ['User_Count'])


def get_group_label(x, groups=None):
    if groups is None:
        return "Other"
    else:
        for key, val in groups.items():
            if x in val:
                return key
        return "Other"


platforms = {"Playstation": ["PS", "PS2", "PS3", "PS4"],
             "Xbox": ["XB", "X360", "XOne"],
             "PC": ["PC"],
             "Nintendo": ["Wii", "WiiU"],
             "Portable": ["GB", "GBA", "GC", "DS", "3DS", "PSP", "PSV"]}

data["Grouped_Platform"] = data["Platform"].apply(lambda x: get_group_label(x, groups=platforms))

scored['Grouped_Platform'] = scored['Platform'].apply(lambda x: get_group_label(x, platforms))

scored['Weighted_Score'] = (scored['User_Score'] * scored['User_Count'] * 10 + scored['Critic_Score'] * scored[
    'Critic_Count']) / (scored['User_Count'] + scored['Critic_Count'])

devs = pd.DataFrame(
    {'dev': scored['Developer'].value_counts().index, 'count': scored['Developer'].value_counts().values})
mean_score = pd.DataFrame({'dev': scored.groupby('Developer')['Weighted_Score'].mean().index,
                           'Mean_score': scored.groupby('Developer')['Weighted_Score'].mean().values})

devs = pd.merge(devs, mean_score, on='dev')
devs = devs.sort_values(by='count', ascending=False)  # I don't think this is necessary
devs["percent"] = devs["count"] / devs["count"].sum()
devs["top%"] = devs["percent"].cumsum() * 100
n_groups = 5
devs["top_group"] = (devs["top%"] * n_groups)
devs["top_group"].iloc[-1] = n_groups

data['Critic_Score'].fillna(0.0, inplace=True)
data['Critic_Count'].fillna(0.0, inplace=True)
data['User_Score'].fillna(0.0, inplace=True)
data['User_Count'].fillna(0.0, inplace=True)
data = data.join(devs.set_index('dev')['top_group'], on='Developer')
data = data.rename(columns={'top_group': 'Developer_Rank'})
data['Developer_Rank'].fillna(0.0, inplace=True)
data['Rating'].fillna('None', inplace=True)

temp, rmvd_temp = rm_outlier(data[data['User_Count'] != 0], ['User_Count'])
data.drop(rmvd_temp.index, axis=0, inplace=True)

data['Weighted_Score'] = (data['User_Count'] * data['User_Score'] * 10 + data['Critic_Count'] * data[
    'Critic_Score']) / (data['Critic_Count'] + data['User_Count'])
data['Weighted_Score'].fillna(0.0, inplace=True)

numeric_subset = data.select_dtypes('number').drop(columns=['NA', 'EU', 'JP', 'Other', 'Year'])
categorical_subset = data[['Grouped_Platform', 'Genre', 'Rating']]

mapping = []
for column in categorical_subset.columns:
    weighted_score_median = scored.groupby(column).median()['Weighted_Score']
    mapping.append({'column': column, 'mapping': [x for x in np.argsort(weighted_score_median).items()]})

encoder = ce.ordinal.OrdinalEncoder()
categorical_subset = encoder.fit_transform(categorical_subset, mapping=mapping)

features = pd.concat([numeric_subset, categorical_subset], axis=1)
correlations = features.corr()['Global'].dropna().sort_values()

target = pd.Series(features['Global'])
features = features.drop(columns=['Global'])

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2,
                                                                            random_state=42)


def model_eval(model):
    model.fit(features_train, target_train)
    prediction = model.predict(features_test)
    mae = mean_absolute_error(target_test.tolist(), prediction)
    rmse = sqrt(mean_squared_error(target_test.tolist(), prediction))
    return mae, rmse


# Linear Regression
lr = LinearRegression()
lrmae, lrrmse = model_eval(lr)

# SVM
SVM = SVR()
svmmae, svmrmse = model_eval(SVM)

# Random Forest
rf = RandomForestRegressor()
rfmae, rfrmse = model_eval(rf)

# Gradient Boosting Regression
gbr = GradientBoostingRegressor()
gbrmae, gbrrmse = model_eval(gbr)

# Ridge Regression
rr = Ridge()
rrmae, rrrmse = model_eval(rr)

# knn
knn = KNeighborsRegressor(n_neighbors=10)
knnmae, knnrmse = model_eval(knn)

model_comparison_mae = pd.DataFrame({'Model': ['Linear Regression', 'SVM',
                                               'Random Forest', 'Grad Boost', 'Ridge', 'KNN'],
                                     'MAE': [lrmae, svmmae, rfmae, gbrmae, rrmae, knnmae]})

model_comparison_rmse = pd.DataFrame({'Model': ['Linear Regression', 'SVM',
                                                'Random Forest', 'Grad Boost', 'Ridge', 'KNN'],
                                      'RMSE': [lrrmse, svmrmse, rfrmse, gbrrmse, rrrmse, knnrmse]})

loss = ['ls', 'lad', 'huber']

max_depth = [2, 3, 5, 10, 15]

min_samples_leaf = [1, 2, 4, 6, 8]

min_samples_split = [2, 4, 6, 10]

max_features = ['auto', 'sqrt', 'log2', None]

hyperparameter_grid = {'loss': loss, 'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf,
                       'min_samples_split': min_samples_split,
                       'max_features': max_features}

model = GradientBoostingRegressor(random_state=42)

randomcv = RandomizedSearchCV(estimator=model, param_distributions=hyperparameter_grid, cv=2, n_iter=20,
                              scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1, return_train_score=True,
                              random_state=42)

# randomcv.fit(features_train, target_train)
# pickle.dump(randomcv, open('randomcv.sav', 'wb'))

randomcv = pickle.load(open('randomcv.sav', 'rb'))

trees_grid = {'n_estimators': [50, 100, 150, 200, 250, 300]}
gridcv = GridSearchCV(estimator=randomcv.best_estimator_, param_grid=trees_grid, cv=4,
                      scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1, return_train_score=True)

# gridcv.fit(features_train, target_train)
# pickle.dump(gridcv, open('gridcv.sav', 'wb'))

gridcv = pickle.load(open('gridcv.sav', 'rb'))

# results = pd.DataFrame(gridcv.cv_results_)
# Graphs.error_graph(results)

final_model = gridcv.best_estimator_
prediction = final_model.predict(features_test)
error = sqrt(mean_squared_error(target_test, prediction))
# print('The RMSE value is: {:.02f}'.format(error))


random_test = features_test.sample()
prediction = final_model.predict(random_test)
random_case = data.loc[random_test.index]
random_case = random_case.drop(
    columns=['Age', 'Country', 'Has_Score', 'Grouped_Platform', 'Developer_Rank', 'Weighted_Score'])
random_case = random_case.reset_index(drop=True)
temp = float(''.join(str(x) for x in prediction))
random_case['Prediction(M)'] = '{:.04f}'.format(temp)
col_list = random_case.columns.values.tolist()


def get_values():
    return target_train, target_test, prediction, model_comparison_mae, model_comparison_rmse, random_case, col_list
