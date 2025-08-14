"""Modüler özellik mühendisliği (v2)

Bu dosya veri sözlüğüne uygun olarak üç ana sınıf sağlar:
- UserFeatureEngineer
- ProductFeatureEngineer
- InteractionFeatureEngineer

Her sınıf kendi veri kaynağını kabul eder ve `transform()` metoduyla bir
DataFrame döndürür. Tüm sınıflar `user_id` ve/veya `product_id` anahtar
kolonlarını kullanarak birleştirilebilir.
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Optional
import pyarrow.parquet as pq


class InteractionFeatureEngineer:
    """Oturum verisinden (train_sessions.parquet) özellik üretir."""

    def __init__(self, df: pd.DataFrame, cutoff_time: Optional[pd.Timestamp] = None):
        self.df = df.copy()
        self.cutoff_time = cutoff_time
        self._standardize_columns()

    def _standardize_columns(self):
        self.df = self.df.rename(
            columns={
                "ts_hour": "event_time",
                "user_id_hashed": "user_id",
                "content_id_hashed": "product_id",
            }
        )

    def transform(self) -> pd.DataFrame:
        df = self.df
        # Zaman sızıntısını engelle: cutoff_time sonrası kayıtları çıkar
        if self.cutoff_time is not None and "event_time" in df.columns:
            df = df[df["event_time"] < self.cutoff_time]
        # Basit toplulaştırma: kullanıcı + ürün bazlı etkileşim sayıları
        agg = (
            df.groupby(["user_id", "product_id"]).agg(
                clicks_sum=("clicked", "sum") if "clicked" in df.columns else ("product_id", "size"),
                orders_sum=("ordered", "sum") if "ordered" in df.columns else ("product_id", "size"),
                carts_sum=("added_to_cart", "sum") if "added_to_cart" in df.columns else ("product_id", "size"),
                favs_sum=("favourited", "sum") if "favourited" in df.columns else ("product_id", "size"),
                first_interaction=("event_time", "min") if "event_time" in df.columns else ("product_id", "size"),
                last_interaction=("event_time", "max") if "event_time" in df.columns else ("product_id", "size"),
            )
        )
        agg = agg.reset_index()
        agg["total_sessions"] = (
            agg[["clicks_sum", "orders_sum", "carts_sum", "favs_sum"]].sum(axis=1)
        )

        # Ürün popülerliği ve trend (cutoff öncesi pencereler)
        if "event_time" in df.columns:
            cutoff = self.cutoff_time if self.cutoff_time is not None else df["event_time"].max()
            for days in [7, 14, 28]:
                start = cutoff - pd.Timedelta(days=days)
                win = df[(df["event_time"] >= start) & (df["event_time"] < cutoff)]
                pop = win.groupby("product_id").size().rename(f"product_pop_{days}d").reset_index()
                agg = agg.merge(pop, on="product_id", how="left")
                agg[f"product_pop_{days}d"] = agg[f"product_pop_{days}d"].fillna(0).astype("int64")
            # Trend: 7d / 28d oranı
            if "product_pop_7d" in agg.columns and "product_pop_28d" in agg.columns:
                agg["product_pop_trend_7_28"] = (
                    agg["product_pop_7d"] / agg["product_pop_28d"].replace(0, 1)
                )

            # Yakın dönem kullanıcı ve ürün yoğunluğu (1/3/7 gün)
            for d in [1, 3, 7]:
                start = cutoff - pd.Timedelta(days=d)
                win = df[(df["event_time"] >= start) & (df["event_time"] < cutoff)]
                # user bazlı
                ucnt = win.groupby("user_id").size().rename(f"user_recent_clicks_{d}d").reset_index()
                agg = agg.merge(ucnt, on="user_id", how="left")
                agg[f"user_recent_clicks_{d}d"] = agg[f"user_recent_clicks_{d}d"].fillna(0).astype("int64")
                # product bazlı
                pcnt = win.groupby("product_id").size().rename(f"product_recent_clicks_{d}d").reset_index()
                agg = agg.merge(pcnt, on="product_id", how="left")
                agg[f"product_recent_clicks_{d}d"] = agg[f"product_recent_clicks_{d}d"].fillna(0).astype("int64")

            # Recency (saat): cutoff - last_interaction
            if "last_interaction" in agg.columns:
                # güvenli dönüştürme
                try:
                    rec = (cutoff - pd.to_datetime(agg["last_interaction"])) / pd.Timedelta(hours=1)
                    agg["recency_hours"] = rec.clip(lower=0).astype(float)
                except Exception:
                    pass

        # Click/order oranları (varsa)
        for a, b, name in [
            ("clicks_sum", "total_sessions", "click_rate"),
            ("orders_sum", "total_sessions", "order_rate"),
        ]:
            if a in agg.columns and b in agg.columns:
                agg[name] = agg[a] / agg[b].replace(0, 1)

        return agg


class UserFeatureEngineer:
    """user/metadata.parquet ve/veya user log dosyalarından kullanıcı özellikleri."""

    def __init__(self, meta_df: pd.DataFrame, log_df: Optional[pd.DataFrame] = None):
        self.meta_df = meta_df.rename(columns={"user_id_hashed": "user_id"}).copy()
        if log_df is not None:
            self.log_df = log_df.rename(columns={"user_id_hashed": "user_id", "ts_hour": "event_time"})
        else:
            self.log_df = None

    def transform(self) -> pd.DataFrame:
        features = self.meta_df.copy()
        # Yaş tahmini
        if "user_birth_year" in features.columns:
            features["user_age"] = 2025 - features["user_birth_year"].fillna(0)
        # Tenure gruplama
        if "user_tenure_in_days" in features.columns:
            features["tenure_months"] = features["user_tenure_in_days"] / 30

        if self.log_df is not None:
            stats = (
                self.log_df.groupby("user_id").agg(
                    total_search_imp=("total_search_impression", "sum"),
                    total_search_click=("total_search_click", "sum"),
                    last_search=("event_time", "max"),
                )
            ).reset_index()
            features = features.merge(stats, on="user_id", how="left")
        return features


class ProductFeatureEngineer:
    """Ürün metadata + fiyat/geçmiş dosyalarından özellikler üretir."""

    def __init__(
        self,
        meta_df: pd.DataFrame,
        price_df: Optional[pd.DataFrame] = None,
        search_df: Optional[pd.DataFrame] = None,
    ):
        self.meta_df = meta_df.rename(columns={"content_id_hashed": "product_id"}).copy()
        if price_df is not None:
            self.price_df = price_df.rename(columns={"content_id_hashed": "product_id"})
        else:
            self.price_df = None
        if search_df is not None:
            self.search_df = search_df.rename(columns={"content_id_hashed": "product_id"})
        else:
            self.search_df = None

    def transform(self) -> pd.DataFrame:
        features = self.meta_df.copy()
        # Basit kategori kodlamaları
        for col in ["level1_category_name", "level2_category_name", "leaf_category_name"]:
            if col in features.columns:
                features[col] = features[col].astype("category").cat.codes

        if self.price_df is not None:
            price_stats = (
                self.price_df.groupby("product_id").agg(
                    price_min=("discounted_price", "min"),
                    price_max=("discounted_price", "max"),
                    price_mean=("discounted_price", "mean"),
                    review_count=("content_review_count", "max"),
                    rate_avg=("content_rate_avg", "max"),
                )
            ).reset_index()
            features = features.merge(price_stats, on="product_id", how="left")

        if self.search_df is not None:
            search_stats = (
                self.search_df.groupby("product_id").agg(
                    total_search_imp=("total_search_impression", "sum"),
                    total_search_click=("total_search_click", "sum"),
                )
            ).reset_index()
            features = features.merge(search_stats, on="product_id", how="left")

        return features


# Yardımcı fonksiyon: dosyaları oku ve full feature set hazırla

def build_full_feature_set(project_root: Path, data_root: Optional[Path] = None, cutoff_time: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Tüm özellik setini oluşturur.

    data_root verilirse Kaggle özgün dizin yapısı beklenir:
      - train_sessions.parquet
      - content/metadata.parquet
      - content/price_rate_review_data.parquet (opsiyonel)
      - content/search_log.parquet (opsiyonel)
      - user/metadata.parquet
      - user/search_log.parquet (opsiyonel)

    data_root verilmezse proje içindeki raw_data/ altındaki
    'train_data.parquet', 'user_data.parquet', 'product_data.parquet' okunur.
    """

    if data_root is None:
        raw = project_root / "raw_data"
        # --- Etkileşim ---
        inter_df = pd.read_parquet(raw / "train_data.parquet")
        inter_features = InteractionFeatureEngineer(inter_df, cutoff_time=cutoff_time).transform()

        # --- Kullanıcı ---
        user_meta = pd.read_parquet(raw / "user_data.parquet")
        user_log = None
        user_features = UserFeatureEngineer(user_meta, user_log).transform()

        # --- Ürün ---
        prod_meta = pd.read_parquet(raw / "product_data.parquet")
        product_features = ProductFeatureEngineer(prod_meta).transform()
    else:
        # Kaggle dizin yerleşimi
        dr = Path(data_root)

        # --- Etkileşim ---
        inter_path = dr / "train_sessions.parquet"
        inter_df = pd.read_parquet(inter_path)
        inter_features = InteractionFeatureEngineer(inter_df, cutoff_time=cutoff_time).transform()

        # -------------- Yardımcılar --------------
        def _chunked_read_aggregate(parquet_path: Path, time_col: str, cols: list, start_ts: pd.Timestamp, end_ts: pd.Timestamp, freq: str = '1D', rename_map: Optional[dict] = None, tz_localize_utc: bool = False, group_keys: list = None, sum_cols: list = None) -> Optional[pd.DataFrame]:
            """
            Büyük parquet dosyalarını haftalık dilimlerle okuyup her dilimi hemen agregatlayarak belleği düşük tut.
            """
            if not parquet_path.exists():
                return None
            schema = pq.read_schema(parquet_path)
            schema_names = set(schema.names)
            present = [c for c in cols if c in schema_names]
            if time_col not in present:
                # zaman kolonu yoksa işlenemez
                return None
            present = list(dict.fromkeys(present))  # unique ve sırayı koru
            # time_col'un tz bilgisini al ve start/end'i buna göre normalize et
            tz_info = None
            try:
                tz_info = getattr(schema.field(time_col).type, 'tz', None)
            except Exception:
                tz_info = None

            def _norm(ts: pd.Timestamp) -> pd.Timestamp:
                try:
                    if tz_info is None:
                        # parquet kolonu tz-naive ise: tz bilgisi varsa UTC'ye çevirip tz'siz yap
                        if getattr(ts, 'tzinfo', None) is not None:
                            try:
                                return ts.tz_convert('UTC').tz_localize(None)
                            except Exception:
                                return ts.tz_localize(None)
                        return ts
                    else:
                        # parquet kolonu tz-aware ise: ts'i o tz'ye çevir
                        if getattr(ts, 'tzinfo', None) is None:
                            # naive ise UTC varsay ve hedef tz'ye çevir
                            return ts.tz_localize('UTC').tz_convert(tz_info)
                        else:
                            return ts.tz_convert(tz_info)
                except Exception:
                    return ts

            start_f = _norm(start_ts)
            end_f = _norm(end_ts)
            results = []
            cursor = start_f
            while cursor < end_f:
                chunk_end = min(cursor + pd.Timedelta(freq), end_f)
                tbl = pq.read_table(
                    parquet_path,
                    columns=present,
                    filters=[(time_col, '>=', cursor), (time_col, '<', chunk_end)]
                )
                if tbl.num_rows == 0:
                    cursor = chunk_end
                    continue
                dfc = tbl.to_pandas(types_mapper=None)
                # şema uyarlama
                if rename_map:
                    dfc = dfc.rename(columns={k: v for k, v in rename_map.items() if k in dfc.columns})
                # zaman tipi/UTC
                if tz_localize_utc and 'event_time' in dfc.columns:
                    if not pd.api.types.is_datetime64_any_dtype(dfc['event_time']):
                        dfc['event_time'] = pd.to_datetime(dfc['event_time'], errors='coerce')
                    try:
                        if getattr(dfc['event_time'].dt, 'tz', None) is None:
                            dfc['event_time'] = dfc['event_time'].dt.tz_localize('UTC')
                    except Exception:
                        pass
                # event_time varsa sadece NaN temizle; aralık filtresi pyarrow ile yapıldı
                if 'event_time' in dfc.columns:
                    dfc = dfc.dropna(subset=['event_time'])
                # agregat
                if group_keys and sum_cols:
                    have = [c for c in sum_cols if c in dfc.columns]
                    if len(have) == 0:
                        cursor = chunk_end
                        continue
                    grp_keys = [g for g in group_keys if g in dfc.columns]
                    if len(grp_keys) == 0:
                        cursor = chunk_end
                        continue
                    ag = dfc.groupby(grp_keys, as_index=False)[have].sum()
                    results.append(ag)
                else:
                    results.append(dfc)
                del dfc, tbl
                cursor = chunk_end
            if not results:
                return None
            out = pd.concat(results, ignore_index=True)
            return out
        user_meta_path = dr / "user" / "metadata.parquet"
        user_meta = pd.read_parquet(user_meta_path)
        user_log = None
        user_search_log = dr / "user" / "search_log.parquet"
        if user_search_log.exists():
            user_log = pd.read_parquet(user_search_log)
        user_features = UserFeatureEngineer(user_meta, user_log).transform()

        # --- Ürün ---
        prod_meta_path = dr / "content" / "metadata.parquet"
        prod_meta = pd.read_parquet(prod_meta_path)
        price_path = dr / "content" / "price_rate_review_data.parquet"
        price_df = pd.read_parquet(price_path) if price_path.exists() else None
        search_path = dr / "content" / "search_log.parquet"
        search_df = pd.read_parquet(search_path) if search_path.exists() else None
        product_features = ProductFeatureEngineer(prod_meta, price_df=price_df, search_df=search_df).transform()

    # --- Birleştir ---
    df = inter_features.merge(user_features, on="user_id", how="left")
    if "product_id" in product_features.columns:
        df = df.merge(product_features, on="product_id", how="left")
    else:
        print("⚠️  product_id sütunu bulunamadı, ürün özellikleri atlandı.")
    
    # Kategori kimliği türet (metadata isimlerinden)
    if "category_id" not in df.columns:
        for cand in ["leaf_category_name", "level2_category_name", "level1_category_name"]:
            if cand in df.columns:
                df["category_id"] = df[cand]
                print(f"ℹ️  category_id '{cand}' üzerinden türetildi.")
                break

    # Eğitim uyumluluğu: zaman ve hedef kolonları
    if "event_time" not in df.columns and "last_interaction" in df.columns:
        df["event_time"] = df["last_interaction"]
    # Binary hedefler
    if "ordered" not in df.columns and "orders_sum" in df.columns:
        df["ordered"] = (df["orders_sum"].fillna(0) > 0).astype(int)
    if "clicked" not in df.columns and "clicks_sum" in df.columns:
        df["clicked"] = (df["clicks_sum"].fillna(0) > 0).astype(int)

    # Recency ve zaman özellikleri (cutoff_time varsa)
    try:
        # recency (saat): cutoff - last_interaction
        if "last_interaction" in df.columns:
            # cutoff_time erişimi: build_full_feature_set içinden closure ile gelemez; bu nedenle
            # last_interaction'a dayalı zaman özelliklerini yine de türetelim.
            # recency_hours'ı yalnızca event_time varsa ve cutoff_time argümanı ile geldiğinde Interaction tarafında hesaplamak idealdir.
            # Burada event_time düzeyinde değiliz; last_interaction üzerinden saat ve gün çıkarımı yapalım.
            if pd.api.types.is_datetime64_any_dtype(df["last_interaction"]):
                df["last_interaction_hour"] = df["last_interaction"].dt.hour
                df["last_interaction_dow"] = df["last_interaction"].dt.dayofweek
                df["last_interaction_is_weekend"] = df["last_interaction_dow"].isin([5, 6]).astype(int)
    except Exception as e:
        print("⚠️  recency/zaman özellikleri hesaplanamadı:", e)

    # Kategori popülerliği (cutoff öncesi tıklama toplamına dayalı, leakage yok)
    try:
        if "category_id" in df.columns and "clicks_sum" in df.columns:
            cat_pop = (
                df.groupby("category_id")["clicks_sum"].sum().rename("category_pop_all").reset_index()
            )
            df = df.merge(cat_pop, on="category_id", how="left")
            # Normalize (0-1) için basit ölçekleme
            max_pop = df["category_pop_all"].max()
            if pd.notnull(max_pop) and max_pop > 0:
                df["category_pop_all_norm"] = df["category_pop_all"] / max_pop
    except Exception as e:
        print("⚠️  kategori popülerliği hesaplanamadı:", e)

    # Kategori popülerliği pencereleri (7/14/28g) ve trend (cutoff_time kullanarak, sızıntısız)
    try:
        if data_root is not None and cutoff_time is not None and "category_id" in df.columns:
            sessions = pd.read_parquet(data_root / "train_sessions.parquet", columns=None)
            sessions = sessions.rename(columns={
                "ts_hour": "event_time",
                "content_id_hashed": "product_id",
                "user_id_hashed": "user_id",
            })
            # product_id -> category_id mapping
            prod_cat = (
                df[["product_id", "category_id"]]
                .dropna()
                .drop_duplicates()
            )
            sessions = sessions.merge(prod_cat, on="product_id", how="left")
            sessions = sessions.dropna(subset=["event_time", "category_id"])  # güvenlik
            sessions = sessions[sessions["event_time"] < cutoff_time]
            for days in [7, 14, 28]:
                start = cutoff_time - pd.Timedelta(days=days)
                win = sessions[(sessions["event_time"] >= start) & (sessions["event_time"] < cutoff_time)]
                cpop = win.groupby("category_id").size().rename(f"category_pop_{days}d").reset_index()
                df = df.merge(cpop, on="category_id", how="left")
                df[f"category_pop_{days}d"] = df[f"category_pop_{days}d"].fillna(0).astype("int64")
            if "category_pop_7d" in df.columns and "category_pop_28d" in df.columns:
                df["category_pop_trend_7_28"] = (
                    df["category_pop_7d"] / df["category_pop_28d"].replace(0, 1)
                )

            # Yakın dönem kullanıcı ve ürün yoğunluğu (1/3/7 gün)
            for d in [1, 3, 7]:
                start = cutoff_time - pd.Timedelta(days=d)
                win = sessions[(sessions["event_time"] >= start) & (sessions["event_time"] < cutoff_time)]
                # user bazlı
                ucnt = win.groupby("user_id").size().rename(f"user_recent_clicks_{d}d").reset_index()
                df = df.merge(ucnt, on="user_id", how="left")
                df[f"user_recent_clicks_{d}d"] = df[f"user_recent_clicks_{d}d"].fillna(0).astype("int64")
                # product bazlı
                pcnt = win.groupby("product_id").size().rename(f"product_recent_clicks_{d}d").reset_index()
                df = df.merge(pcnt, on="product_id", how="left")
                df[f"product_recent_clicks_{d}d"] = df[f"product_recent_clicks_{d}d"].fillna(0).astype("int64")

            # Recency (saat): cutoff - last_interaction
            if "last_interaction" in df.columns:
                # güvenli dönüştürme
                try:
                    rec = (cutoff_time - pd.to_datetime(df["last_interaction"])) / pd.Timedelta(hours=1)
                    df["recency_hours"] = rec.clip(lower=0).astype(float)
                except Exception:
                    pass

        # Click/order oranları (varsa)
        for a, b, name in [
            ("clicks_sum", "total_sessions", "click_rate"),
            ("orders_sum", "total_sessions", "order_rate"),
        ]:
            if a in df.columns and b in df.columns:
                df[name] = df[a] / df[b].replace(0, 1)

        # Fashion search log (user/fashion_search_log.parquet) agregatları
    except Exception as e:
        print("⚠️  kategori pencereleri/recency oranları hesaplanamadı:", e)

    # Fashion search log (user/fashion_search_log.parquet) agregatları
        try:
            if data_root is not None and cutoff_time is not None:
                fs_path = data_root / "user" / "fashion_search_log.parquet"
                if fs_path.exists():
                    cu = cutoff_time.tz_convert('UTC') if getattr(cutoff_time, 'tzinfo', None) is not None else cutoff_time
                    start28 = cu - pd.Timedelta(days=28)
                    cols = ['ts_hour','user_id_hashed','content_id_hashed','total_search_impression','total_search_click']
                    # Pencere bazlı doğrudan agregasyon (her pencere için ayrı okuma)
                    for days in [7, 14, 28]:
                        start_win = cu - pd.Timedelta(days=days)
                        fsw = _chunked_read_aggregate(
                            fs_path,
                            time_col='ts_hour',
                            cols=cols,
                            start_ts=start_win.tz_localize(None) if getattr(start_win,'tzinfo',None) is not None else start_win,
                            end_ts=cu.tz_localize(None) if getattr(cu,'tzinfo',None) is not None else cu,
                            freq='7D',
                            rename_map={'ts_hour':'event_time','user_id_hashed':'user_id','content_id_hashed':'product_id',
                                        'total_search_impression':'search_impression','total_search_click':'search_click'},
                            tz_localize_utc=False,
                            group_keys=['user_id','product_id'],
                            sum_cols=['search_impression','search_click']
                        )
                        if fsw is not None:
                            ren = {}
                            if 'search_impression' in fsw.columns:
                                ren['search_impression'] = f"user_csearch_impr_{days}d"
                            if 'search_click' in fsw.columns:
                                ren['search_click'] = f"user_csearch_clk_{days}d"
                            fsw = fsw.rename(columns=ren)
                            df = df.merge(fsw, on=['user_id','product_id'], how='left')
        except Exception as e:
            print("⚠️  fashion_search_log agregatları hesaplanamadı:", e)

        # content/sitewide_log.parquet -> ürün bazlı sitewide click/cart/fav/order pencereleri
        try:
            if data_root is not None and cutoff_time is not None:
                sw_path = data_root / "content" / "sitewide_log.parquet"
                if sw_path.exists():
                    cu = cutoff_time.tz_convert('UTC') if getattr(cutoff_time, 'tzinfo', None) is not None else cutoff_time
                    start28 = cu - pd.Timedelta(days=28)
                    cols = ['date','content_id_hashed','total_click','total_cart','total_fav','total_order']
                    # Her pencere için doğrudan agregasyon
                    for days in [7, 14, 28]:
                        start_win = cu - pd.Timedelta(days=days)
                        sw = _chunked_read_aggregate(
                            sw_path,
                            time_col='date',
                            cols=cols,
                            start_ts=start_win.tz_localize(None) if getattr(start_win,'tzinfo',None) is not None else start_win,
                            end_ts=cu.tz_localize(None) if getattr(cu,'tzinfo',None) is not None else cu,
                            freq='7D',
                            rename_map={'date':'event_time','content_id_hashed':'product_id'},
                            tz_localize_utc=False,
                            group_keys=['product_id'],
                            sum_cols=['total_click','total_cart','total_fav','total_order']
                        )
                        if sw is not None:
                            grp = sw.rename(columns={
                                'total_click': f"product_site_clicks_{days}d",
                                'total_cart': f"product_site_carts_{days}d",
                                'total_fav': f"product_site_favs_{days}d",
                                'total_order': f"product_site_orders_{days}d",
                            })
                            df = df.merge(grp, on=['product_id'], how='left')
                        # CTR benzeri
                        if 'product_site_total_click_28d' in df.columns and 'product_site_total_order_28d' in df.columns:
                            denom = (df['product_site_total_click_28d'] + 1e-9)
                            df['product_site_order_per_click_28d'] = df['product_site_total_order_28d'] / denom
        except Exception as e:
            print("⚠️  sitewide_log agregatları hesaplanamadı:", e)

        # content/search_log.parquet -> ürün bazlı arama impression/click pencereleri
        try:
            if data_root is not None and cutoff_time is not None:
                csl_path = data_root / "content" / "search_log.parquet"
                if csl_path.exists():
                    cu = cutoff_time.tz_convert('UTC') if getattr(cutoff_time, 'tzinfo', None) is not None else cutoff_time
                    start28 = cu - pd.Timedelta(days=28)
                    cols = ['date','content_id_hashed','total_search_impression','total_search_click']
                    for days in [7, 14, 28]:
                        start_win = cu - pd.Timedelta(days=days)
                        csl = _chunked_read_aggregate(
                            csl_path,
                            time_col='date',
                            cols=cols,
                            start_ts=start_win.tz_localize(None) if getattr(start_win,'tzinfo',None) is not None else start_win,
                            end_ts=cu.tz_localize(None) if getattr(cu,'tzinfo',None) is not None else cu,
                            freq='7D',
                            rename_map={'date':'event_time','content_id_hashed':'product_id'},
                            tz_localize_utc=False,
                            group_keys=['product_id'],
                            sum_cols=['total_search_impression','total_search_click']
                        )
                        if csl is not None:
                            grp = csl.rename(columns={
                                'total_search_impression': f"csearch_impr_{days}d",
                                'total_search_click': f"csearch_clk_{days}d",
                            })
                            df = df.merge(grp, on=['product_id'], how='left')
                        if 'csearch_clk_28d' in df.columns and 'csearch_impr_28d' in df.columns:
                            denom = df['csearch_impr_28d'].replace(0,1)
                            df['csearch_ctr_28d'] = df['csearch_clk_28d'] / denom
        except Exception as e:
            print("⚠️  content/search_log agregatları hesaplanamadı:", e)

        # content/top_terms_log.parquet -> ürün × terim bazlı toplu arama; ürün toplamı alınır
        try:
            if data_root is not None and cutoff_time is not None:
                ttl_path = data_root / "content" / "top_terms_log.parquet"
                if ttl_path.exists():
                    cu = cutoff_time.tz_convert('UTC') if getattr(cutoff_time, 'tzinfo', None) is not None else cutoff_time
                    start28 = cu - pd.Timedelta(days=28)
                    cols = ['date','content_id_hashed','total_search_impression','total_search_click','search_term_normalized']
                    for days in [7, 14, 28]:
                        start_win = cu - pd.Timedelta(days=days)
                        ttl = _chunked_read_aggregate(
                            ttl_path,
                            time_col='date',
                            cols=cols,
                            start_ts=start_win.tz_localize(None) if getattr(start_win,'tzinfo',None) is not None else start_win,
                            end_ts=cu.tz_localize(None) if getattr(cu,'tzinfo',None) is not None else cu,
                            freq='7D',
                            rename_map={'date':'event_time','content_id_hashed':'product_id'},
                            tz_localize_utc=False,
                            group_keys=['product_id','search_term_normalized'] if 'search_term_normalized' in cols else ['product_id'],
                            sum_cols=['total_search_impression','total_search_click']
                        )
                        if ttl is not None:
                            cols_avail = [c for c in ["total_search_impression","total_search_click"] if c in ttl.columns]
                            if cols_avail:
                                grp = ttl.groupby('product_id', as_index=False)[cols_avail].sum()
                                rename = {}
                                if 'total_search_impression' in cols_avail:
                                    rename['total_search_impression'] = f"product_term_impr_{days}d"
                                if 'total_search_click' in cols_avail:
                                    rename['total_search_click'] = f"product_term_clk_{days}d"
                                grp = grp.rename(columns=rename)
                                df = df.merge(grp, on='product_id', how='left')
                        if 'product_term_clk_28d' in df.columns and 'product_term_impr_28d' in df.columns:
                            denom = df['product_term_impr_28d'].replace(0,1)
                            df['product_term_ctr_28d'] = df['product_term_clk_28d'] / denom
        except Exception as e:
            print("⚠️  top_terms_log agregatları hesaplanamadı:", e)

    # user/fashion_sitewide_log.parquet -> kullanıcı bazlı sitewide click/cart/fav/order pencereleri
    try:
        if data_root is not None and cutoff_time is not None:
            usw_path = data_root / "user" / "fashion_sitewide_log.parquet"
            if usw_path.exists():
                cu = cutoff_time.tz_convert('UTC') if getattr(cutoff_time, 'tzinfo', None) is not None else cutoff_time
                start28 = cu - pd.Timedelta(days=28)
                cols = ['ts_hour','user_id_hashed','total_click','total_cart','total_fav','total_order']
                schema_names = set(pq.read_schema(usw_path).names)
                present = [c for c in cols if c in schema_names]
                usw = pq.read_table(
                    usw_path,
                    columns=present,
                    filters=[('ts_hour','>=', start28), ('ts_hour','<', cu)]
                ).to_pandas()
                usw = usw.rename(columns={
                    'ts_hour': 'event_time',
                    'user_id_hashed': 'user_id'
                })
                needed = ['event_time','user_id']
                usw_cols = [c for c in ['total_click','total_cart','total_fav','total_order'] if c in usw.columns]
                if all(c in usw.columns for c in needed) and len(usw_cols)>0:
                    usw = usw.dropna(subset=['event_time'])
                    usw = usw[usw['event_time'] < cutoff_time]
                    for days in [7,14,28]:
                        start = cutoff_time - pd.Timedelta(days=days)
                        win = usw[(usw['event_time'] >= start) & (usw['event_time'] < cutoff_time)]
                        for m in usw_cols:
                            agg = win.groupby('user_id')[m].sum().rename(f"user_site_{m}_{days}d").reset_index()
                            df = df.merge(agg, on='user_id', how='left')
                    if 'user_site_total_click_28d' in df.columns and 'user_site_total_order_28d' in df.columns:
                        denom = (df['user_site_total_click_28d'] + 1e-9)
                        df['user_site_order_per_click_28d'] = df['user_site_total_order_28d'] / denom
                else:
                    print("ℹ️  fashion_sitewide_log beklenen kolonlar yok, atlandı.")
    except Exception as e:
        print("⚠️  fashion_sitewide_log agregatları hesaplanamadı:", e)

    return df
