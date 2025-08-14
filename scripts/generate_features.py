"""Generate full feature set for Trendyol Datathon 2025.

Çalıştırmak için:
    python scripts/generate_features.py

Bu script şunları yapar:
1. raw_data klasöründen train_data.parquet + user_data.parquet + product_data.parquet dosyalarını okur.
2. FeatureEngineer sınıfını kullanarak ürün, kullanıcı, etkileşim ve seans özelliklerini çıkarır.
3. Tüm özellikleri tek bir DataFrame'de birleştirir ve `data/training_set_features.parquet` dosyasına yazar.
"""
from __future__ import annotations

import pandas as pd
import os
import sys
from pathlib import Path

# Proje kökü
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "raw_data"
FEATURE_DIR = PROJECT_ROOT / "data"
FEATURE_DIR.mkdir(exist_ok=True)

sys.path.append(str(PROJECT_ROOT / "scripts"))
from feature_engineering import FeatureEngineer  # noqa: E402


def main():
    print("📊 Ham veriler yükleniyor …")

    train_path = RAW_DIR / "train_data.parquet"
    user_path = RAW_DIR / "user_data.parquet"
    product_path = RAW_DIR / "product_data.parquet"

    interaction_df = pd.read_parquet(train_path)
    # Kolon adlarını standardize et
    interaction_df = interaction_df.rename(columns={
        'content_id_hashed': 'product_id',
        'user_id_hashed': 'user_id',
        'ts_hour': 'event_time'
    })
    user_df = pd.read_parquet(user_path)
    user_df = user_df.rename(columns={'user_id_hashed': 'user_id'})
    product_df = pd.read_parquet(product_path)

    print("Ham veriler yüklendi:")
    print(f"   train:   {interaction_df.shape}")
    print(f"   users:   {user_df.shape}")
    print(f"   products:{product_df.shape}")

    engineer = FeatureEngineer()

    # Ürün veri setinde beklenen sütunlar (ör. product_id) yoksa özelliği atla
    if 'content_id_hashed' in product_df.columns:
        print("🔧 Ürün özellikleri çıkarılıyor …")
        product_features = engineer.create_product_features(product_df.rename(columns={'content_id_hashed': 'product_id'}),
                                                           interaction_df.rename(columns={'content_id_hashed': 'product_id'}))
    else:
        print("⚠️  Ürün bazlı özellikler atlandı (uygun sütun bulunamadı).")
        product_features = pd.DataFrame()

    print("🔧 Kullanıcı özellikleri çıkarılıyor …")
    user_features = engineer.create_user_features(user_df, interaction_df)

    print("🔧 Etkileşim özellikleri birleştiriliyor …")
    interaction_features = engineer.create_interaction_features(
        interaction_df, product_features, user_features
    )

    # Opsiyonel: session_features (varsa event_time)
    # interaction_features = engineer.create_session_features(interaction_features)

    # Kaydet
    features_path = FEATURE_DIR / "training_set_features.parquet"
    interaction_features.to_parquet(features_path, index=False)
    print(f"✅ Özellik seti kaydedildi: {features_path}")


if __name__ == "__main__":
    main()
