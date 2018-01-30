import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class FeatureExtractor():

    def __init__(self):
        pass

    def fit(self, X_df, y=None):
        pass

    def transform(self, X_df):
        X_df = X_df.copy()
        X_df.drop(['Latitude', 'Longitude', '# of Docks'],
                  axis=1, inplace=True)

        X_df['time'] = pd.to_datetime(X_df['day'] + ' ' + X_df['timestamp'])
        X_df.drop(['day', 'timestamp'], axis=1, inplace=True)

        ohe_day = OneHotEncoder()
        ohe_hour = OneHotEncoder()

        weekday_unq = np.unique(X_df['weekday'])
        week_day = X_df['weekday'].values
        ohe_day.fit(weekday_unq.reshape(-1, 1))
        week_day_ohe_sparse = ohe_day.transform(week_day.reshape(-1, 1))

        hour_unq = np.unique(X_df['hour'])
        hours = X_df['weekday'].values
        ohe_hour.fit(hour_unq.reshape(-1, 1))
        hours_ohe_sparse = ohe_hour.transform(hours.reshape(-1, 1))

        temp = X_df['temp'].values
        rain_fall = X_df['precip'].values

        return np.c_[week_day_ohe_sparse.todense(),
                     hours_ohe_sparse.todense(),
                     temp, rain_fall]
