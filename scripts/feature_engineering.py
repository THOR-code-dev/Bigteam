"""
Trendyol Datathon 2025 - Feature Engineering

Bu modül, ürün, kullanıcı ve etkileşim özelliklerini oluşturmak için kullanılır.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Feature engineering class for Trendyol Datathon 2025
    """
    
    def __init__(self):
        self.product_features = None
        self.user_features = None
        self.interaction_features = None
        self.session_features = None
    
    def create_product_features(self, product_df: pd.DataFrame, 
                              interaction_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create product-level features
        
        Args:
            product_df: Product information dataframe
            interaction_df: Interaction data for calculating statistics
        
        Returns:
            DataFrame with product features
        """
        # Ensure product_id exists; if not, derive from interaction_df
        if 'product_id' not in product_df.columns:
            features = pd.DataFrame({'product_id': interaction_df['product_id'].unique()})
        else:
            features = product_df.copy()
        
        # Basic product info
        if 'price' in features.columns:
            features['price_log'] = np.log1p(features['price'])
            features['price_category_ratio'] = features['price'] / features.groupby('category_id')['price'].transform('mean')
        else:
            # Fiyat bilgisi yoksa varsayılan sütunları 0 olarak ekle
            features['price_log'] = 0.0
            features['price_category_ratio'] = 0.0
        
        # Interaction-based features
        if interaction_df is not None:
            # Click statistics
            click_stats = interaction_df.groupby('product_id').agg({
                'clicked': ['sum', 'mean', 'count'],
                'ordered': ['sum', 'mean']
            }).reset_index()
            click_stats.columns = ['product_id', 'total_clicks', 'click_rate', 'total_impressions', 
                                 'total_orders', 'order_rate']
            
            # Conversion rate
            click_stats['conversion_rate'] = click_stats['total_orders'] / (click_stats['total_clicks'] + 1)
            click_stats['click_through_rate'] = click_stats['total_clicks'] / (click_stats['total_impressions'] + 1)
            
            features = features.merge(click_stats, on='product_id', how='left')
            features.fillna(0, inplace=True)
        
        return features
    
    def create_user_features(self, user_df: pd.DataFrame, 
                           interaction_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create user-level features
        
        Args:
            user_df: User information dataframe
            interaction_df: Interaction data for calculating statistics
        
        Returns:
            DataFrame with user features
        """
        features = user_df.copy()
        
        if interaction_df is not None:
            # User behavior statistics
            agg_dict = {
                'session_id': 'nunique',
                'clicked': ['sum', 'mean'],
                'ordered': ['sum', 'mean'],
            }
            if 'price' in interaction_df.columns:
                agg_dict['price'] = ['mean', 'max', 'min']
            user_stats = interaction_df.groupby('user_id').agg(agg_dict)
            # Flatten Çoklu index isimleri
            user_stats.columns = ['_'.join([str(c) for c in col if c]).strip('_') for col in user_stats.columns]
            user_stats = user_stats.reset_index()
            # yeni isimleri değişkenlere ata
            # emniyet: eksik kolonları doldur
            expected_cols = ['session_id_nunique', 'clicked_sum', 'clicked_mean', 'ordered_sum', 'ordered_mean']
            for col in expected_cols:
                if col not in user_stats.columns:
                    user_stats[col] = 0
            if 'price_mean' not in user_stats.columns:
                user_stats['price_mean'] = 0
                user_stats['price_max'] = 0
                user_stats['price_min'] = 0
            # Kolonları daha açıklayıcı isimlere çevir
            user_stats = user_stats.rename(columns={
                'session_id_nunique': 'total_sessions',
                'clicked_sum': 'total_clicks',
                'clicked_mean': 'avg_click_rate',
                'ordered_sum': 'total_orders',
                'ordered_mean': 'avg_order_rate',
                'price_mean': 'avg_price',
                'price_max': 'max_price',
                'price_min': 'min_price'
            })

            # Ek oranlar
            user_stats['orders_per_session'] = user_stats['total_orders'] / (user_stats['total_sessions'] + 1)
            user_stats['clicks_per_session'] = user_stats['total_clicks'] / (user_stats['total_sessions'] + 1)
            if 'max_price' in user_stats.columns:
                user_stats['price_range'] = user_stats['max_price'] - user_stats['min_price']
            else:
                user_stats['price_range'] = 0.0

            user_stats['clicks_per_session'] = user_stats['total_clicks'] / (user_stats['total_sessions'] + 1)
            user_stats['price_range'] = user_stats['max_price'] - user_stats['min_price']
            
            features = features.merge(user_stats, on='user_id', how='left')
            features.fillna(0, inplace=True)
        
        return features
    
    def create_interaction_features(self, interaction_df: pd.DataFrame,
                                  product_features: pd.DataFrame,
                                  user_features: pd.DataFrame) -> pd.DataFrame:
        """
        Create user-product interaction features
        
        Args:
            interaction_df: Interaction dataframe
            product_features: Product features dataframe
            user_features: User features dataframe
        
        Returns:
            DataFrame with interaction features
        """
        features = interaction_df.copy()
        
        # Merge product and user features
        if product_features is not None:
            features = features.merge(product_features, on='product_id', how='left', suffixes=('', '_product'))
        
        if user_features is not None:
            features = features.merge(user_features, on='user_id', how='left', suffixes=('', '_user'))
        
        # User-product specific features
        if interaction_df is not None:
            # Historical interaction count
            user_product_history = interaction_df.groupby(['user_id', 'product_id']).size().reset_index(name='user_product_interactions')
            features = features.merge(user_product_history, on=['user_id', 'product_id'], how='left')
            features['user_product_interactions'].fillna(0, inplace=True)
            
            # Category preferences
            user_category_stats = interaction_df.groupby(['user_id', 'category_id']).agg({
                'clicked': 'sum',
                'ordered': 'sum'
            }).reset_index()
            user_category_stats['category_preference'] = user_category_stats['clicked'] + user_category_stats['ordered']
            
            features = features.merge(user_category_stats[['user_id', 'category_id', 'category_preference']], 
                                    on=['user_id', 'category_id'], how='left')
            features['category_preference'].fillna(0, inplace=True)
        
        return features
    
    def create_session_features(self, session_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create session-level features
        
        Args:
            session_df: Session data dataframe
        
        Returns:
            DataFrame with session features
        """
        features = session_df.copy()
        
        # Session statistics
        session_stats = session_df.groupby('session_id').agg({
            'product_id': 'count',
            'price': ['mean', 'std', 'min', 'max']
        }).reset_index()
        session_stats.columns = ['session_id', 'products_in_session', 'avg_session_price', 
                               'std_session_price', 'min_session_price', 'max_session_price']
        
        # Price range in session
        session_stats['price_range_session'] = session_stats['max_session_price'] - session_stats['min_session_price']
        
        features = features.merge(session_stats, on='session_id', how='left')
        
        return features
    
    def save_features(self, features: Dict[str, pd.DataFrame], output_dir: str):
        """
        Save features to parquet files
        
        Args:
            features: Dictionary of feature dataframes
            output_dir: Directory to save features
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        for name, df in features.items():
            output_path = f"{output_dir}/{name}.parquet"
            df.to_parquet(output_path, index=False)
            print(f"Saved {name} features to {output_path}")

if __name__ == "__main__":
    # Example usage
    engineer = FeatureEngineer()
    
    # Create sample data
    products = pd.DataFrame({
        'product_id': [1, 2, 3],
        'price': [100, 200, 150],
        'category_id': ['A', 'B', 'A'],
        'brand_id': ['X', 'Y', 'X']
    })
    
    users = pd.DataFrame({
        'user_id': [1, 2, 3],
        'age': [25, 30, 35]
    })
    
    interactions = pd.DataFrame({
        'user_id': [1, 1, 2, 2, 3],
        'product_id': [1, 2, 2, 3, 1],
        'session_id': ['s1', 's1', 's2', 's2', 's3'],
        'clicked': [1, 0, 1, 1, 0],
        'ordered': [0, 0, 0, 1, 0],
        'price': [100, 200, 200, 150, 100],
        'category_id': ['A', 'B', 'B', 'A', 'A']
    })
    
    # Create features
    product_features = engineer.create_product_features(products, interactions)
    user_features = engineer.create_user_features(users, interactions)
    interaction_features = engineer.create_interaction_features(interactions, product_features, user_features)
    
    print("Feature engineering completed successfully!")
