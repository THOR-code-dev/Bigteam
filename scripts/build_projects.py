from pathlib import Path
import pandas as pd
from feature_engineering_v2 import build_full_feature_set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(r"E:/Bigteam/Project/data")

# Cutoff_time'ı her zaman RAW etkileşim verisinden hesapla (birikimli küçülmeyi önlemek için)
def compute_cutoff_from_raw(data_root: Path) -> pd.Timestamp | None:
    try:
        inter_path = data_root / "train_sessions.parquet"
        df = pd.read_parquet(inter_path, columns=None)
        # Sütun standartlaştırma
        if "ts_hour" in df.columns and "event_time" not in df.columns:
            df = df.rename(columns={"ts_hour": "event_time"})
        if "event_time" not in df.columns:
            print("[WARN] RAW'da event_time/ts_hour bulunamadı; cutoff hesaplanamadı.")
            return None
        if len(df) == 0:
            print("[WARN] RAW etkileşim verisi boş; cutoff hesaplanamadı.")
            return None
        split_idx = int(len(df) * 0.8)
        cutoff = df["event_time"].sort_values().iloc[split_idx]
        print("[CUT] cutoff_time (RAW 80% split):", cutoff)
        return cutoff
    except Exception as e:
        print("[WARN] RAW'dan cutoff hesaplanamadı:", e)
        return None

out_path = PROJECT_ROOT / "data" / "training_set_features.parquet"
cutoff_time = compute_cutoff_from_raw(DATA_ROOT)

df = build_full_feature_set(PROJECT_ROOT, data_root=DATA_ROOT, cutoff_time=cutoff_time)
out_path.parent.mkdir(exist_ok=True)
df.to_parquet(out_path, index=False)
print("[OK] Özellik seti kaydedildi:", out_path)
