import pandas as pd
from matplotlib import pyplot as plt
import scikitplot as skplt


def plot_data_distribution(features: pd.DataFrame):
    plt.figure(figsize=(32, 24))
    features.plot(kind="hist", legend=None, bins=20, color='k')
    features.plot(kind="kde", legend=None)


def plot_learning_curve(model, X_train, y_train, cv, title, scoring):
    skplt.estimators.plot_learning_curve(model, X_train, y_train, cv=cv,
                                         shuffle=True, scoring=scoring,
                                         n_jobs=-1, figsize=(16, 12),
                                         title_fontsize="large", text_fontsize="large",
                                         title=title)


def plot_confusion_matrix(y_test, pred, title):
    skplt.metrics.plot_confusion_matrix(y_test, pred, title=title, figsize=(16, 12),
                                        cmap='Paired', text_fontsize='large')


def plot_roc_auc_curve(y_test, pred_prob, title):
    skplt.metrics.plot_roc(y_test, pred_prob, title=title, plot_micro=False, figsize=(16, 12), cmap='Paired')


def plot_precision_recall_curve(y_test, pred_prob, title):
    skplt.metrics.plot_precision_recall(y_test, pred_prob, title=title, figsize=(16, 12),
                                        cmap='Paired', plot_micro=False)
