"""
Trendyol Datathon 2025 - Model Training Script

Bu script, LightGBM model eğitimi ve optimizasyonu için kullanılır.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
# Zaman bazlı split fonksiyonu (aynı klasörden)
from validation import time_based_split
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import argparse
import optuna
import os
import json
import sys
from pathlib import Path

# Proje kök dizinini ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.metrics import calculate_recall_at_k, calculate_weighted_recall

class ModelTrainer:
    """
    LightGBM model trainer for Trendyol Datathon 2025
    """
    
    def __init__(self):
        self.model = None
        self.features = None
        self.best_params = None
        
    def load_data(self, features_path: str, target_col: str = 'clicked'):
        """
        Load training data
        
        Args:
            features_path: Path to features file
            target_col: Target column name
        """
        data = pd.read_parquet(features_path)
        
        # Separate features and target
        feature_cols = [col for col in data.columns if col not in ['session_id', 'user_id', 'product_id', 'clicked', 'ordered']]
        
        X = data[feature_cols]
        y = data[target_col]
        
        return X, y, data
    
    def train_baseline_model(self, X_train, y_train, X_val, y_val):
        """
        Train baseline LightGBM model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
        """
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )
        
        return self.model
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Features to predict on
            
        Returns:
            predictions: Model predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_baseline_model first.")
        
        return self.model.predict(X)

    def train_with_params(self, X_train, y_train, X_val, y_val, params: dict):
        """
        Train LightGBM with provided params and return the model.
        """
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=2000,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
        )
        return self.model
    
    def save_model(self, model_path: str):
        """
        Save trained model
        
        Args:
            model_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """
        Load trained model
        
        Args:
            model_path: Path to load the model from
        """
        self.model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")

if __name__ == "__main__":
    # Example usage
    trainer = ModelTrainer()
    
    # Paths
    BASE_DIR = Path(__file__).resolve().parents[1]
    TRAIN_FEATURES_PATH = str(BASE_DIR / 'data' / 'training_set_features.parquet')
    MODEL_PATH = str(BASE_DIR / 'models' / 'baseline_model.pkl')
    
    # Load data
    print("Loading training data...")
    X, y, data = trainer.load_data(TRAIN_FEATURES_PATH)
    
            # Zaman bazlı eğitim/validasyon ayırımı
    target_col = "clicked"
    X_train, y_train, X_val, y_val = time_based_split(
        data,
        time_col="event_time",
        val_ratio=0.2,
        feature_cols_exclude=["session_id", "user_id", "product_id"],
        target_col=target_col,
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Label mean (train): {y_train.mean():.6f}, Label mean (val): {y_val.mean():.6f}")
    
    # Yalnızca sayısal özelliklerle eğit (LightGBM gereksinimi)
    numeric_cols = X_train.select_dtypes(include=[np.number, bool]).columns.tolist()
    # Kaçak bilgi (leakage) içeren kolonları hariç tut
    leakage_cols = [
        'clicks_sum', 'orders_sum', 'carts_sum', 'favs_sum',
        'total_sessions', 'click_rate', 'order_rate'
    ]
    numeric_cols = [c for c in numeric_cols if c not in leakage_cols]
    X_train = X_train[numeric_cols].fillna(0)
    X_val = X_val[numeric_cols].fillna(0)
    print(f"Kullanılan özellik sayısı (numeric, no-leak): {len(numeric_cols)}")
    
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpo", action="store_true", help="Run Optuna HPO before final training")
    parser.add_argument("--trials", type=int, default=15, help="Number of HPO trials")
    parser.add_argument("--ranker", action="store_true", help="Train LightGBM Ranker (lambdarank) grouped by user_id")
    args = parser.parse_args()

    baseline_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }

    if args.hpo and not args.ranker:
        print(f"Running HPO with {args.trials} trials...")

        def objective(trial: optuna.trial.Trial):
            params = baseline_params.copy()
            params.update({
                'num_leaves': trial.suggest_int('num_leaves', 16, 256),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 200),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 5.0),
                'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 5.0),
                'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 1.0),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            })
            try:
                model = trainer.train_with_params(X_train, y_train, X_val, y_val, params)
                val_pred = model.predict(X_val)
                auc = roc_auc_score(y_val, val_pred)
                return auc
            except Exception as e:
                # Invalid config -> very low score
                return 0.0

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=args.trials)
        best_params = baseline_params.copy()
        best_params.update(study.best_params)
        print("Best params:", best_params)
        print(f"Best AUC: {study.best_value:.6f}")
        print("Training final model with best params...")
        model = trainer.train_with_params(X_train, y_train, X_val, y_val, best_params)
    elif args.hpo and args.ranker:
        # Ranker HPO (lambdarank) - ndcg@20 maksimize
        print(f"Running Ranker HPO (lambdarank) with {args.trials} trials...")
        # Grup uzunluklarını hazırla
        train_users = data.loc[X_train.index, "user_id"].values
        val_users = data.loc[X_val.index, "user_id"].values
        def to_group_lengths(u):
            import numpy as np
            _, counts = np.unique(u, return_counts=True)
            return counts.tolist()
        g_train = to_group_lengths(train_users)
        g_val = to_group_lengths(val_users)

        dtrain_full = lgb.Dataset(X_train, label=y_train, group=g_train)
        dval_full = lgb.Dataset(X_val, label=y_val, group=g_val, reference=dtrain_full)

        def objective_ranker(trial: optuna.trial.Trial):
            params = {
                'objective': 'lambdarank',
                'metric': 'ndcg',
                'ndcg_at': [20],
                'boosting_type': 'gbdt',
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 31, 255),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 500),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 0, 10),
                'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 5.0),
                'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 5.0),
                'max_depth': trial.suggest_int('max_depth', -1, 16),
                'verbose': -1,
                'seed': 42,
            }
            model = lgb.train(
                params,
                dtrain_full,
                valid_sets=[dval_full],
                num_boost_round=2000,
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
            )
            return model.best_score.get('valid_0', {}).get('ndcg@20', float('nan'))

        study = optuna.create_study(direction='maximize')
        study.optimize(objective_ranker, n_trials=args.trials, show_progress_bar=False)
        best_params = study.best_params
        # Sabitleri geri ekle
        best_params.update({
            'objective': 'lambdarank', 'metric': 'ndcg', 'ndcg_at': [20], 'boosting_type': 'gbdt', 'verbose': -1, 'seed': 42
        })
        print("Best Ranker params:", best_params)
        print(f"Best valid ndcg@20: {study.best_value:.6f}")

        # En iyi parametrelerle final eğitimi
        model = lgb.train(
            best_params,
            dtrain_full,
            valid_sets=[dval_full],
            num_boost_round=2000,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
        )
        trainer.model = model
    else:
        if args.ranker:
            # Ranker eğitimi (lambdarank) - kullanıcı bazlı gruplama
            print("Training LightGBM Ranker (lambdarank)...")
            # Grup dizileri: her kullanıcı için satır sayısı
            train_users = data.loc[X_train.index, "user_id"].values
            val_users = data.loc[X_val.index, "user_id"].values
            def to_group_lengths(u):
                # u sıralı değilse sıralı indexe göre grupla
                import numpy as np
                _, counts = np.unique(u, return_counts=True)
                return counts.tolist()
            g_train = to_group_lengths(train_users)
            g_val = to_group_lengths(val_users)

            ranker_params = {
                'objective': 'lambdarank',
                'metric': 'ndcg',
                'ndcg_at': [20],
                'boosting_type': 'gbdt',
                'num_leaves': 63,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'seed': 42
            }
            dtrain = lgb.Dataset(X_train, label=y_train, group=g_train)
            dval = lgb.Dataset(X_val, label=y_val, group=g_val, reference=dtrain)
            model = lgb.train(
                ranker_params,
                dtrain,
                valid_sets=[dval],
                num_boost_round=2000,
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
            )
            trainer.model = model
        else:
            # Train baseline
            print("Training baseline model...")
            model = trainer.train_baseline_model(X_train, y_train, X_val, y_val)
    
    # Valid AUC raporu
    val_pred = model.predict(X_val)
    try:
        auc = roc_auc_score(y_val, val_pred)
        print(f"Validation AUC: {auc:.6f}")
    except Exception as e:
        print("AUC hesaplanamadı:", e)
    
    # Recall@K (K=20) hesapla: kullanıcı bazında top-K tahmin ve gerçekler
    try:
        K = 20
        # Orijinal dataframe 'data' içinden validation satırlarını yakala
        # X_val index'i, time_based_split çıktılarına göre val_df index'idir
        val_indices = X_val.index
        cols = ["user_id", "product_id", target_col]
        if "ordered" in data.columns:
            cols.append("ordered")
        val_view = data.loc[val_indices, cols].copy()
        val_view["score"] = val_pred
        # Kullanıcı bazında sıralama ve recall@K
        recalls = []
        for uid, grp in val_view.groupby("user_id"):
            grp_sorted = grp.sort_values("score", ascending=False)
            predicted = grp_sorted.head(K)["product_id"].tolist()
            actual = grp.loc[grp[target_col] == 1, "product_id"].tolist()
            if len(actual) == 0:
                continue
            r = calculate_recall_at_k(predicted, actual, K)
            recalls.append(r)
        if len(recalls) > 0:
            print(f"Validation Recall@{K}: {np.mean(recalls):.6f} (n_users_with_labels={len(recalls)})")
        else:
            print("Validation Recall@K hesaplanamadı: valid sette pozitif etiketli kullanıcı bulunamadı.")
    except Exception as e:
        print("Recall@K hesaplanamadı:", e)

    # Kullanıcı-başı AUC (click ve order) ve yarışma skoru (0.3*click + 0.7*order)
    try:
        def per_user_auc_mean(df_view: pd.DataFrame, label_col: str, score_col: str) -> float:
            aucs = []
            for uid, g in df_view.groupby("user_id"):
                y = g[label_col].astype(int).values
                if y.sum() == 0 or y.sum() == len(y):
                    continue  # AUC tanımsız
                s = g[score_col].values
                try:
                    aucs.append(roc_auc_score(y, s))
                except Exception:
                    pass
            return float(np.mean(aucs)) if len(aucs) > 0 else float("nan")

        auc_click_user = per_user_auc_mean(val_view, target_col, "score")
        print(f"Per-user AUC (click): {auc_click_user:.6f}")
        if "ordered" in val_view.columns:
            auc_order_user = per_user_auc_mean(val_view, "ordered", "score")
            print(f"Per-user AUC (order): {auc_order_user:.6f}")
            if not np.isnan(auc_click_user) and not np.isnan(auc_order_user):
                comp_score = 0.3 * auc_click_user + 0.7 * auc_order_user
                print(f"Competition Score (est.): {comp_score:.6f} = 0.3*click + 0.7*order")
        else:
            auc_order_user = float("nan")
            print("Per-user AUC (order): N/A (ordered kolonu yok)")

        # Kaydet: valid tahminleri ve özet metrikleri
        base_dir = Path(__file__).resolve().parents[1]
        out_pred = base_dir / "data" / "validation_predictions.parquet"
        out_metrics = base_dir / "data" / "validation_metrics.json"
        out_pred.parent.mkdir(parents=True, exist_ok=True)
        try:
            val_view.to_parquet(out_pred, index=False)
            metrics = {
                "validation_auc": float(auc) if 'auc' in locals() else None,
                "validation_recall@20": float(np.mean(recalls)) if 'recalls' in locals() and len(recalls)>0 else None,
                "per_user_auc_click": float(auc_click_user),
                "per_user_auc_order": float(auc_order_user),
            }
            # competition score varsa ekle
            if "comp_score" in locals():
                metrics["competition_score_estimate"] = float(comp_score)
            with open(out_metrics, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            print(f"Saved validation predictions to {out_pred}")
            print(f"Saved validation metrics to {out_metrics}")
        except Exception as e:
            print("Validation tahminleri/metrikleri kaydedilemedi:", e)
    except Exception as e:
        print("Per-user AUC ve skor hesaplanamadı:", e)

    # Feature importance
    try:
        importances = pd.Series(model.feature_importance(), index=numeric_cols).sort_values(ascending=False)
        print("Top 20 feature importances:")
        print(importances.head(20))
    except Exception as e:
        print("Feature importance hesaplanamadı:", e)

    # Save model
    # Klasörü oluştur ve modeli kaydet
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    trainer.save_model(MODEL_PATH)
    
    print("Training completed!")
