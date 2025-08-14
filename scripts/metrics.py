"""
Trendyol Datathon 2025 - Evaluation Metrics

Bu modül, yarışmanın değerlendirme metriği olan Recall@K fonksiyonlarını içerir.
"""

import numpy as np
from typing import List, Union

def calculate_recall_at_k(predicted: List[Union[int, str]], 
                         actual: List[Union[int, str]], 
                         k: int = 10) -> float:
    """
    Calculate Recall@K metric for a single session.
    
    Args:
        predicted: List of predicted product IDs in ranked order
        actual: List of actual product IDs user interacted with (clicked/ordered)
        k: Number of top predictions to consider (default: 10)
    
    Returns:
        recall: Recall@K score between 0 and 1
    """
    if not actual or not predicted:
        return 0.0
    
    # Take top k predictions
    predicted_k = predicted[:k]
    
    # Calculate hits (intersection between predicted and actual)
    actual_set = set(actual)
    predicted_set = set(predicted_k)
    hits = len(actual_set.intersection(predicted_set))
    
    return hits / len(actual)

def calculate_weighted_recall(predicted: List[Union[int, str]], 
                            actual_clicked: List[Union[int, str]], 
                            actual_ordered: List[Union[int, str]], 
                            k: int = 10,
                            click_weight: float = 0.1,
                            order_weight: float = 0.9) -> float:
    """
    Calculate weighted Recall@K combining clicks and orders.
    
    Args:
        predicted: List of predicted product IDs in ranked order
        actual_clicked: List of actual clicked product IDs
        actual_ordered: List of actual ordered product IDs
        k: Number of top predictions to consider
        click_weight: Weight for click recall (default: 0.1)
        order_weight: Weight for order recall (default: 0.9)
    
    Returns:
        weighted_recall: Weighted Recall@K score
    """
    click_recall = calculate_recall_at_k(predicted, actual_clicked, k)
    order_recall = calculate_recall_at_k(predicted, actual_ordered, k)
    
    return click_weight * click_recall + order_weight * order_recall

def evaluate_model(predictions_df, actual_df, k_values=[10, 20, 50]):
    """
    Evaluate model performance across multiple K values.
    
    Args:
        predictions_df: DataFrame with session_id and predicted product_ids
        actual_df: DataFrame with session_id, clicked_products, ordered_products
        k_values: List of K values to evaluate
    
    Returns:
        dict: Dictionary with recall scores for each K
    """
    results = {}
    
    for k in k_values:
        total_score = 0
        session_count = 0
        
        for session_id in predictions_df['session_id'].unique():
            pred_products = predictions_df[predictions_df['session_id'] == session_id]['product_id'].tolist()
            
            actual_clicked = actual_df[actual_df['session_id'] == session_id]['clicked_products'].iloc[0] if len(actual_df[actual_df['session_id'] == session_id]) > 0 else []
            actual_ordered = actual_df[actual_df['session_id'] == session_id]['ordered_products'].iloc[0] if len(actual_df[actual_df['session_id'] == session_id]) > 0 else []
            
            score = calculate_weighted_recall(pred_products, actual_clicked, actual_ordered, k)
            total_score += score
            session_count += 1
        
        results[f'recall@{k}'] = total_score / session_count if session_count > 0 else 0
    
    return results

if __name__ == "__main__":
    # Test the functions
    predicted = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    actual_clicked = [2, 4, 6]
    actual_ordered = [1, 3]
    
    print("Recall@10:", calculate_recall_at_k(predicted, actual_clicked + actual_ordered, 10))
    print("Weighted Recall@10:", calculate_weighted_recall(predicted, actual_clicked, actual_ordered, 10))
