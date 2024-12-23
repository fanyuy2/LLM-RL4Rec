import numpy as np


def mean_average_precision(y_true, y_pred):
    """Calculate Mean Average Precision (MAP)."""
    y_true_set = set(y_true)
    avg_precisions = []

    for k in range(1, len(y_pred) + 1):
        if y_pred[k - 1] in y_true_set:
            avg_precisions.append(1.0)
        else:
            avg_precisions.append(0.0)

    return np.mean(avg_precisions)


def ndcg_at_k(y_true, y_pred, k):
    """Calculate Normalized Discounted Cumulative Gain (NDCG) at rank k."""
    if k > len(y_pred):
        k = len(y_pred)

    dcg = 0.0
    for i in range(k):
        if y_pred[i] in y_true:
            dcg += 1 / np.log2(i + 2)  # Log base 2, starting from 1 (i + 2)

    idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(y_true))))  # Ideal DCG
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(y_true, y_pred, k):
    """Calculate Precision at rank k."""
    if k > len(y_pred):
        k = len(y_pred)

    relevant_items = sum(1 for item in y_pred[:k] if item in y_true)
    return relevant_items / k if k > 0 else 0.0


def recall_at_k(y_true, y_pred, k):
    """Calculate Recall at rank k."""
    if k > len(y_pred):
        k = len(y_pred)

    relevant_items = sum(1 for item in y_pred[:k] if item in y_true)
    return relevant_items / len(y_true) if len(y_true) > 0 else 0.0


def evaluate_recommendations(y_true, y_pred, k):
    """Evaluate the recommendations using multiple metrics."""
    metrics = {
        "MAP": mean_average_precision(y_true, y_pred),
        "NDCG@k": ndcg_at_k(y_true, y_pred, k),
        "Precision@k": precision_at_k(y_true, y_pred, k),
        "Recall@k": recall_at_k(y_true, y_pred, k),
    }
    return metrics



def evaluate_recommendations_for_all_users(y_true_all, y_pred_all, top_k):
    map_scores = []
    ndcg_scores = []
    precision_scores = []
    recall_scores = []

    for user_id, y_pred in y_pred_all.items():
        y_true = y_true_all[user_id]
        map_scores.append(mean_average_precision(y_true, y_pred))
        ndcg_scores.append(ndcg_at_k(y_true, y_pred, top_k))
        precision_scores.append(precision_at_k(y_true, y_pred, top_k))
        recall_scores.append(recall_at_k(y_true, y_pred, top_k))

    # # Compute metrics for each user
    # for y_true, y_pred in zip(y_true_all, y_pred_all):
    #     map_scores.append(mean_average_precision(y_true, y_pred))
    #     ndcg_scores.append(ndcg_at_k(y_true, y_pred, k))
    #     precision_scores.append(precision_at_k(y_true, y_pred, k))
    #     recall_scores.append(recall_at_k(y_true, y_pred, k))


    # Average metrics across all users
    metrics = {
        "MAP": np.mean(map_scores),
        "NDCG@k": np.mean(ndcg_scores),
        "Precision@k": np.mean(precision_scores),
        "Recall@k": np.mean(recall_scores),
    }
    return metrics
