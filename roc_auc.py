from sklearn.metrics import roc_auc_score

def compute_auc(y_true,
                y_prob):

    return roc_auc_score(
        y_true,
        y_prob
    )