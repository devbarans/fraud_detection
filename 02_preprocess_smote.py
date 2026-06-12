# =============================================================================
# Adım 2: Veri Ön İşleme ve SMOTE Uygulaması
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def on_isleme_ve_kaydet(veri_yolu: str = "data/creditcard.csv"):
    """
    Veri setini ön işler, eğitim ve test kümelerine böler.
    Veri sızıntısını önlemek amacıyla SMOTE işlemini yalnızca eğitim setine uygular.
    """
    if not os.path.exists(veri_yolu):
        raise FileNotFoundError(f"Hata: '{veri_yolu}' dosyası bulunamadı. Lütfen öncelikle Adım 1 (01_eda.py) modülünü çalıştırınız.")

    df = pd.read_csv(veri_yolu)
    
    # Kopyasını çıkarıp ön işleme başla
    df_copy = df.copy()
    df_copy["Amount_log"] = np.log1p(df_copy["Amount"])
    df_copy.drop(columns=["Amount", "Time"], inplace=True)

    # Amount_log'u normalize et / ölçeklendir
    scaler = StandardScaler()
    df_copy["Amount_log"] = scaler.fit_transform(df_copy[["Amount_log"]])

    X = df_copy.drop(columns=["Class"])
    y = df_copy["Class"]

    # Sınıf dağılımını koruyarak ayır (stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Eğitim seti boyutu : {len(X_train):,} | Test seti boyutu: {len(X_test):,}")

    # SMOTE — YALNIZCA EĞİTİM SETİNE (Veri sızıntısını önler)
    print("\n[INFO] SMOTE öncesi eğitim seti sınıf dağılımı:")
    unique_b, counts_b = np.unique(y_train, return_counts=True)
    for u, c in zip(unique_b, counts_b):
        print(f"  Sınıf {u}: {c:,}")

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print("\n[INFO] SMOTE sonrası eğitim seti sınıf dağılımı:")
    unique_a, counts_a = np.unique(y_train_res, return_counts=True)
    for u, c in zip(unique_a, counts_a):
        print(f"  Sınıf {u}: {c:,}")

    # NumPy İkili (Binary) Dosyaları Olarak Kaydet
    np.save("data/X_train.npy", X_train_res.values)
    np.save("data/X_test.npy", X_test.values)
    np.save("data/y_train.npy", y_train_res.values)
    np.save("data/y_test.npy", y_test.values)
    np.save("data/feature_names.npy", np.array(X.columns.tolist(), dtype=object))

    print("\n[SUCCESS] İşlenmiş veriler başarıyla kaydedildi:")
    print("   data/X_train.npy, data/X_test.npy, data/y_train.npy, data/y_test.npy, data/feature_names.npy")

    # Şekil 6 SMOTE Görselleştirmesini Üret
    sekil6_smote_gorseli(
        X_train.values, y_train.values, 
        X_train_res.values, y_train_res.values, 
        X.columns.tolist()
    )

def sekil6_smote_gorseli(X_train, y_train, X_train_res, y_train_res, ozellik_isimleri):
    """
    SMOTE işleminin sınıf ve boyutsal dağılım (V14 vs V10) üzerindeki etkilerini görselleştirir.
    """
    print("\n[INFO] SMOTE öncesi/sonrası görselleştirmesi (Şekil 6) üretiliyor...")
    
    # Grafik parametrelerini güncelle
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Şekil 6 — SMOTE Öncesi ve Sonrası Veri Dağılım Analizi", fontsize=14, fontweight="bold", y=0.98)

    # ── 1. SOL PANEL: SINIF DAĞILIMI KARŞILAŞTIRMASI (SAYISAL) ──
    counts_before = np.bincount(y_train)
    counts_after = np.bincount(y_train_res)

    labels = ["Meşru\n(Class 0)", "Dolandırıcılık\n(Class 1)"]
    x = np.arange(len(labels))
    width = 0.35

    # Barlar: Premium renk paletiyle uyumlu lacivert ve koyu kırmızı tonları
    rects1 = ax1.bar(x - width/2, counts_before, width, label="SMOTE Öncesi", color=["#4C82B6", "#8E2C2C"], alpha=0.6, edgecolor="black")
    rects2 = ax1.bar(x + width/2, counts_after, width, label="SMOTE Sonrası", color=["#20507a", "#b83a2c"], alpha=0.9, edgecolor="black")

    # Çok büyük fark olduğu için y-eksenini logaritmik yapıyoruz
    ax1.set_yscale("log")
    ax1.set_ylim(10, counts_after.max() * 30)  # Baş payı (headroom) bırakarak gösterge kutusunun sayıların üzerine binmesini engelliyoruz.
    ax1.set_ylabel("Gözlem Sayısı (Log Ölçeği)", fontsize=11)
    ax1.set_title("Eğitim Seti Sınıf Dengelenmesi (Sayısal)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(axis='y', linestyle=':', alpha=0.5)

    # Değerleri barların üzerine ekle
    for rect in rects1:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2, h * 1.2 if h < 1000 else h * 1.05,
                 f"{int(h):,}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#333333")
    for rect in rects2:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2, h * 1.2 if h < 1000 else h * 1.05,
                 f"{int(h):,}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#111111")

    # ── 2. SAĞ PANEL: GEOMETRİK DAĞILIM DEĞİŞİMİ (V14 vs V10) ──
    # En önemli iki özellik olan V14 ve V10 indislerini bulalım
    try:
        v14_col_idx = ozellik_isimleri.index("V14")
        v10_col_idx = ozellik_isimleri.index("V10")
    except ValueError:
        v14_col_idx = 13
        v10_col_idx = 9

    # Çizimde üst üste binmeleri önlemek ve net görmek için rastgele örnekleme
    np.random.seed(42)
    
    # SMOTE Sonrası Dağılım
    c0_idx_a = np.where(y_train_res == 0)[0]
    c1_idx_a = np.where(y_train_res == 1)[0]
    
    c0_sample_a = np.random.choice(c0_idx_a, min(3000, len(c0_idx_a)), replace=False)
    c1_sample_a = np.random.choice(c1_idx_a, min(3000, len(c1_idx_a)), replace=False)

    x_c0_a = X_train_res[c0_sample_a, v14_col_idx]
    y_c0_a = X_train_res[c0_sample_a, v10_col_idx]
    x_c1_a = X_train_res[c1_sample_a, v14_col_idx]
    y_c1_a = X_train_res[c1_sample_a, v10_col_idx]

    # Sağ panel scatter çizimi
    ax2.scatter(x_c0_a, y_c0_a, color="#20507a", alpha=0.3, label="Meşru (Class 0)", s=8)
    ax2.scatter(x_c1_a, y_c1_a, color="#b83a2c", alpha=0.4, label="Sentetik + Gerçek Fraud (Class 1)", s=8)

    ax2.set_xlabel("V14 Bileşeni", fontsize=11)
    ax2.set_ylabel("V10 Bileşeni", fontsize=11)
    ax2.set_title("SMOTE Sonrası Karar Uzayı (V14 vs V10)", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower left", fontsize=10)
    ax2.grid(linestyle=':', alpha=0.5)

    # Odak uzayı netleştirmek için eksenleri sınırla
    ax2.set_xlim(-15, 6)
    ax2.set_ylim(-15, 8)

    plt.tight_layout()
    # figures dizini yoksa oluştur
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/sekil6_smote_gorseli.png", bbox_inches="tight")
    plt.close()
    print("[SUCCESS] Şekil 6 başarıyla oluşturuldu: figures/sekil6_smote_gorseli.png")

if __name__ == "__main__":
    print("=" * 60)
    print("ADIM 2: VERİ ÖN İŞLEME VE SMOTE SÜRECİ BAŞLATILDI")
    print("=" * 60)
    
    on_isleme_ve_kaydet()
    
    print("=" * 60)
    print("ADIM 2: İŞLEMLER BAŞARIYLA TAMAMLANDI")
    print("=" * 60)
