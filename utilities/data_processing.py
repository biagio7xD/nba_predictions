import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler, StandardScaler


def get_robust_scaled_data(features):
    robust_scaler = RobustScaler(with_scaling=True, with_centering=True)
    robust_scaler.fit(features)
    return pd.DataFrame(robust_scaler.transform(features), columns=features.columns, index=features.index)


def get_normalized_features(features):
    tmp = preprocessing.normalize(features)
    return pd.DataFrame(tmp, columns=features.columns, index=features.index)


def get_standard_scaled_data(features):
    robust_scaler = StandardScaler()
    robust_scaler.fit(features)
    return pd.DataFrame(robust_scaler.transform(features), columns=features.columns, index=features.index)
