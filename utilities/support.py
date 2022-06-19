import pandas as pd
from sklearn import metrics
from sklearn.utils import shuffle


def load_data(csv_file, game_date_column: str, label: str):
    nba_df = pd.read_csv(csv_file)
    nba_df[game_date_column] = pd.to_datetime(nba_df[game_date_column], format='%Y-%m-%d')
    nba_df[label] = nba_df[label].astype("category")
    return nba_df


def shuffle_dataset(df: pd.DataFrame):
    return shuffle(df, random_state=42).reset_index(drop=True)


def get_features_labels_from_df(df: pd.DataFrame, cols_to_delete: list, label):
    return df.drop(cols_to_delete, axis=1), df[label]


def print_model_metrics(y_test, pred, pred_prob):
    print(f'F1 Score: {round(metrics.f1_score(y_test, pred), 2)}')
    print(f'Accuracy Score: {round(metrics.accuracy_score(y_test, pred), 2)}')
    print(f'AUC Score: {round(metrics.roc_auc_score(y_test, pred_prob), 3)}')
    print(f'Precision Score: {round(metrics.precision_score(y_test, pred), 3)}')
    print(f'Recall Score: {round(metrics.recall_score(y_test, pred), 3)}')
