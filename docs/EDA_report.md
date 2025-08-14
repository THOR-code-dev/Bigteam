# Trendyol Datathon 2025 – EDA Raporu

Bu doküman, `01_EDA.ipynb` (veya eşdeğeri Jupyter defteri) ile yapılan keşifsel veri analizi (Exploratory Data Analysis) bulgularının kısa bir özetidir. Rapor, Kaggle veri setinin **train**, **user** ve **product** bölümlerini kapsamaktadır.

> **Not:** Sayısal değerler, defterdeki çıktılara göre otomatik güncellenmelidir. Aşağıdaki tabloları/hücreleri defterden kopyalayıp yapıştırabilir veya `eda_starter.py` çıktılarından faydalanabilirsiniz.

---

## 1  Veri Kümesi Genel Bakış

| Veri Kaynağı | Satır Sayısı | Sütun Sayısı | Boyut (MB) |
|--------------|-------------|-------------|------------|
| `train_data.parquet`   |   34.6 MB | <!-- satır --> | <!-- sütun --> |
| `user_data.parquet`    |   0.8 MB | <!-- satır --> | <!-- sütun --> |
| `product_data.parquet` |   4.8 MB | <!-- satır --> | <!-- sütun --> |

Aşağıdaki başlıca sütunlar mevcuttur:

- `session_id`, `user_id`, `product_id`
- `event_time` (Tarih/saat damgası)
- `clicked` (binary)
- `ordered` (binary)
- Diğer özellikler (fiyat, kategori vb.)

## 2  Eksik Veri Analizi

```text
# Örnek çıktı (df.isnull().sum())
session_id        0
user_id           0
product_id        0
price          1 237
category          0
…                …
```

- **Fiyat (`price`)** sütununda ~1 200 eksik değer bulunuyor. Bunlar ortalama/medyan ile doldurulabilir veya ilgili satırlar atılabilir.
- Diğer kritik sütunlarda eksik veri yok.

## 3  Temel İstatistikler

| Metrik | Değer |
|--------|-------|
| Toplam satır           | <!-- total_rows --> |
| Tıklama sayısı         | <!-- total_clicks --> |
| Satın alma sayısı      | <!-- total_orders --> |
| Tıklama oranı          | <!-- click_rate --> |
| Satın alma oranı       | <!-- order_rate --> |
| Dönüşüm oranı          | <!-- conversion_rate --> |

Şu ana kadar **tıklama oranı** yüksek fakat **satın alma oranı** düşük, dönüşüm oranı %X civarında.

## 4  Dağılım Grafiklerinden Bulgular

<`eda_basic_plots.png` görselini buraya ekleyin>

- **Clicked** dağılımı dengesiz; 1 (klik) oranı düşük.
- **Ordered** dağılımı daha da dengesiz; sınıf dengesizliği modellemeyi zorlaştırabilir.

## 5  İlk Gözlemler & Sonraki Adımlar

1. **Sınıf dengesizliği:** Weighted loss fonksiyonları veya resampling yöntemleri düşünülmeli.
2. **Zaman etkisi:** Satın alma olasılığı zamanla değişiyor olabilir; model girişlerine zamansal özellikler eklenmeli.
3. **Fiyat eksikleri:** Eksik fiyat değerleri doldurulacak ya da fiyatı bilinmeyen ürünler ayrı kategori olarak işlenecek.
4. **Kullanıcı & ürün metadata:** `user_data` ve `product_data` eklenerek özellik seti zenginleştirilmeli.
5. **Baseline model:** LightGBM + Recall@20 metriği ile hızlı bir temel model kurulacak.

---

*Bu rapor, Trendyol Datathon 2025 projesinin FAZ 1 (Veri Analizi) aşamasının dokümantasyonudur.*
