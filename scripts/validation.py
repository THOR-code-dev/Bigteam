"""Validation utilities for Trendyol Datathon 2025

Bu modül, zaman bazlı eğitim/validasyon ayırımı (time-based split) sağlar.
Etkileşim verilerinde zamansal sıralamayı korumak önemlidir, bu nedenle
klasik rastgele train_test_split yerine son % val_ratio'luk dilimi
validasyon olarak ayırıyoruz.
"""
from __future__ import annotations

import pandas as pd
from typing import Tuple, List

def time_based_split(
    df: pd.DataFrame,
    time_col: str,
    val_ratio: float = 0.2,
    sort_ascending: bool = True,
    feature_cols_exclude: List[str] | None = None,
    target_col: str = "clicked",
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split dataframe into train/val based on chronological order.

    Parameters
    ----------
    df : pd.DataFrame
        Veri çerçevesi (özellik + hedef) — *tüm satırlar aynı seviye (ör. etkileşim)*.
    time_col : str
        Tarih/zaman bilgisini içeren sütun adı (örn. `event_time`).
    val_ratio : float, default 0.2
        Son *val_ratio* oranındaki veriyi validasyon seti olarak ayırır.
    sort_ascending : bool, default True
        True ise eski → yeni sıralar, False ise tam tersi.
    feature_cols_exclude : list[str] | None
        Özellik listesinden çıkarılacak sütunlar (örn. id ve hedef sütunları).
    target_col : str, default "clicked"
        Hedef (label) sütunu.

    Returns
    -------
    X_train, y_train, X_val, y_val
    """
    if time_col not in df.columns:
        raise ValueError(f"'{time_col}' sütunu dataframe'de bulunamadı.")

    if feature_cols_exclude is None:
        feature_cols_exclude = []

    # Zaman sıralaması
    df_sorted = df.sort_values(time_col, ascending=sort_ascending).reset_index(drop=True)

    split_idx = int(len(df_sorted) * (1 - val_ratio))

    train_df = df_sorted.iloc[:split_idx]
    val_df = df_sorted.iloc[split_idx:]

    feature_cols = [
        col for col in df.columns if col not in feature_cols_exclude + [target_col]
    ]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    return X_train, y_train, X_val, y_val
