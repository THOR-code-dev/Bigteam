"""
Trendyol Datathon 2025 - Submission Creation Script

Bu script, test verisi için tahminler oluşturur ve Kaggle submission formatına dönüştürür.
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys

# Proje kök dizinini ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class SubmissionCreator:
    """
    Create Kaggle submission file
    """
    
    def __init__(self, model_path: str, features_path: str):
        self.model = joblib.load(model_path)
        self.features_path = features_path
        
    def load_test_data(self):
        """Load test features"""
        return pd.read_parquet(self.features_path)
    
    def create_predictions(self, test_df):
        """Create predictions for test data"""
        # Get feature columns
        feature_cols = [col for col in test_df.columns 
                       if col not in ['session_id', 'user_id', 'product_id']]
        
        X_test = test_df[feature_cols]
        
        # Predict probabilities
        predictions = self.model.predict(X_test)
        
        # Add predictions to dataframe
        test_df['prediction_score'] = predictions
        
        return test_df
    
    def create_submission(self, predictions_df, output_path: str):
        """Create submission file in Kaggle format"""
        # Group by session and sort by prediction score
        submission = predictions_df.groupby('session_id').apply(
            lambda x: x.sort_values('prediction_score', ascending=False)['product_id'].tolist()
        ).reset_index()
        
        submission.columns = ['session_id', 'prediction']
        
        # Convert list to string format
        submission['prediction'] = submission['prediction'].apply(
            lambda x: ' '.join(map(str, x))
        )
        
        # Save submission
        submission.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
        
        return submission

if __name__ == "__main__":
    # Paths
    MODEL_PATH = '../models/final_model.pkl'
    TEST_FEATURES_PATH = '../data/test_features.parquet'
    SUBMISSION_PATH = '../submissions/final_submission.csv'
    
    # Create submission
    creator = SubmissionCreator(MODEL_PATH, TEST_FEATURES_PATH)
    
    print("Loading test data...")
    test_df = creator.load_test_data()
    
    print("Creating predictions...")
    predictions_df = creator.create_predictions(test_df)
    
    print("Creating submission file...")
    submission = creator.create_submission(predictions_df, SUBMISSION_PATH)
    
    print(f"Submission shape: {submission.shape}")
    print("Submission completed!")
