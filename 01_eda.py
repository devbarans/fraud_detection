# =============================================================================
# Adım 1: Keşifsel Veri Analizi (EDA)
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Renk Paleti
MAVI    = "#4C72B0"
TURUNCU = "#DD8452"

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

def veriyi_yukle(veri_yolu: str = "data/creditcard.csv") -> pd.DataFrame:
    """
    Veri setini KaggleHub API üzerinden indirir. Dosya mevcutsa indirme işlemini atlar.
    """
    if not os.path.exists(veri_yolu):
        print("[INFO] Kagglehub üzerinden veri seti indiriliyor...")
        try:
            import kagglehub
            import shutil
            downloaded_dir = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
            source_file = os.path.join(downloaded_dir, "creditcard.csv")
            if os.path.exists(source_file):
                shutil.copy(source_file, veri_yolu)
                print("[INFO] Veri seti başarıyla indirildi ve yerel dizine kopyalandı.")
            else:
                files = os.listdir(downloaded_dir)
                for f in files:
                    if f.endswith(".csv"):
                        shutil.copy(os.path.join(downloaded_dir, f), veri_yolu)
                        print(f"[INFO] '{f}' dosyası '{veri_yolu}' olarak başarıyla kopyalandı.")
                        break
        except Exception as e:
            raise SystemExit(
                f"\n[Hata] Veri seti indirme başarısız oldu: {e}\n"
                "Lütfen internet bağlantınızı ve 'kagglehub' kütüphanesinin kurulu olduğunu kontrol ediniz."
            )
    else:
        print(f"[INFO] '{veri_yolu}' dosyası zaten mevcut. İndirme işlemi atlanıyor.")

    df = pd.read_csv(veri_yolu)
    print(f"   [INFO] Veri seti boyutu: {df.shape[0]:,} satır × {df.shape[1]} sütun")
    print(f"   [INFO] Dolandırıcılık (Fraud) gözlem sayısı: {df['Class'].sum():,}  (%{df['Class'].mean()*100:.3f})")
    return df

def sekil1_sinif_dagilimi(df: pd.DataFrame):
    """
    Sınıf dağılımını pasta ve logaritmik sütun grafikleriyle görselleştirir.
    """
    print("\n[INFO] Sınıf dağılımı görselleştiriliyor...")
    counts = df["Class"].value_counts().sort_index()
    # counts[0] = 284315, counts[1] = 492
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Sol: Pasta Grafiği (Pie Chart)
    labels = [f"Meşru İşlem\n({counts[0]:,})", f"Dolandırıcılık\n({counts[1]:,})"]
    colors = ["#20507a", "#b83a2c"]
    
    # Kırmızı dilimi vurgulamak için patlatıyoruz
    explode = (0, 0.35)
    
    wedges, texts, autotexts = ax1.pie(
        counts.values,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct="%1.2f%%",
        startangle=150,
        pctdistance=0.7,
        textprops=dict(fontsize=9, fontweight="bold")
    )
    
    # Metin renklerini ve stillerini ayarla
    texts[0].set_color("#333333")
    texts[1].set_color("#b83a2c")
    autotexts[0].set_color("white")
    autotexts[1].set_color("#b83a2c")
    
    # İkinci sınıfın (Fraud) metnini ve yüzdesini dışarıya konumlandır
    autotexts[1].set_position((-0.8, 0.8))
    
    ax1.set_title("Oransal Dağılım", fontsize=12, fontweight="bold")
    
    # 2. Sağ: Logaritmik Sütun Grafiği
    x_labels = ["Meşru\n(Class 0)", "Dolandırıcılık\n(Class 1)"]
    bars = ax2.bar(x_labels, counts.values, color=colors, width=0.5, edgecolor="white")
    
    # Sütun değerlerini tepelerine yaz
    for bar, val in zip(bars, counts.values):
        offset = val * 1.3 if val < 1000 else val * 1.15
        ax2.text(bar.get_x() + bar.get_width()/2, offset,
                 f"{val:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
                 
    ax2.set_yscale("log")
    ax2.set_ylabel("İşlem Sayısı (log ölçeği)", fontsize=10)
    ax2.set_title("Sayısal Dağılım (Logaritmik Ölçek)", fontsize=12, fontweight="bold")
    ax2.set_ylim(10, counts.max() * 10)
    
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis='y', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("figures/sekil1_sinif_dagilimi.png", bbox_inches="tight")
    plt.close()
    print("[INFO] Şekil 1 kaydedildi: figures/sekil1_sinif_dagilimi.png")

def sekil2_amount_dagilimi(df: pd.DataFrame):
    """
    'Amount' değişkeninin ham ve logaritmik dönüşümlü dağılımlarını karşılaştırır.
    """
    print("\n[INFO] İşlem Tutarı (Amount) dağılımı görselleştiriliyor...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Şekil 2 — Amount Değişkeninin Dağılımı", fontsize=14,
                 fontweight="bold", y=1.02)

    carpiklik_ham = df["Amount"].skew()
    ax1.hist(df["Amount"], bins=100, color=MAVI, alpha=0.75, edgecolor="white")
    ax1.set_title(f"Ham Amount  (çarpıklık: {carpiklik_ham:.2f})", fontsize=11)
    ax1.set_xlabel("Amount")
    ax1.set_ylabel("Frekans")
    ax1.set_yscale("log")

    amount_log = np.log1p(df["Amount"])
    carpiklik_log = amount_log.skew()
    ax2.hist(amount_log, bins=100, color=TURUNCU, alpha=0.75, edgecolor="white")
    ax2.set_title(f"log(Amount + 1)  (çarpıklık: {carpiklik_log:.2f})", fontsize=11)
    ax2.set_xlabel("log(Amount + 1)")
    ax2.set_ylabel("Frekans")

    plt.tight_layout()
    plt.savefig("figures/sekil2_amount_dagilimi.png", bbox_inches="tight")
    plt.close()
    print(f"   [INFO] Çarpıklık derecesi (Skewness): {carpiklik_ham:.2f} → {carpiklik_log:.2f}")
    print("[INFO] Şekil 2 kaydedildi: figures/sekil2_amount_dagilimi.png")

def sekil_boxplot_analizi(df: pd.DataFrame):
    """
    V14, V10, V4 ve Time özellikleri için sınıflara göre kutu grafiklerini çizer.
    """
    print("\n[INFO] V14, V10, V4 ve Time kutu grafikleri çiziliyor...")
    
    degiskenler = ["V14", "V10", "V4", "Time"]
    renkler = ["#20507a", "#b83a2c"]  # Meşru: Lacivert, Dolandırıcılık: Koyu Kırmızı
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Kredi Kartı Dolandırıcılığında Ayırt Edici Bileşenlerin Kutu Grafikleri", 
                 fontsize=14, fontweight="bold", y=0.98)
    
    x_labels = ["Meşru\n(Class 0)", "Dolandırıcılık\n(Class 1)"]
    
    for i, var in enumerate(degiskenler):
        ax = axes[i // 2, i % 2]
        
        # Seaborn boxplot
        sns.boxplot(
            x="Class", 
            y=var, 
            data=df, 
            ax=ax, 
            palette=renkler, 
            hue="Class",
            legend=False,
            width=0.4, 
            showfliers=False,  # Aykırı değerleri gizle ki kutuların medyan ayrışması net görünsün
            linewidth=1.5
        )
        
        # Grid ve etiketler
        ax.set_title(f"{var} Değişkeninin Sınıflara Göre Dağılımı", fontsize=11, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel(var, fontsize=10)
        ax.grid(axis='y', linestyle=':', alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        # Medyan değerlerini bulup grafik üzerine ekle
        medians = df.groupby("Class")[var].median()
        for xtick in ax.get_xticks():
            val = medians[xtick]
            if var == "Time":
                ax.text(xtick, val, f"Med: {val:,.0f}", 
                        horizontalalignment='center', size='small', color='black', weight='bold',
                        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))
            else:
                ax.text(xtick, val, f"Med: {val:.2f}", 
                        horizontalalignment='center', size='small', color='black', weight='bold',
                        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))

    plt.tight_layout()
    # Üst başlığın çakışmaması için pay bırak
    plt.subplots_adjust(top=0.92)
    
    plt.savefig("figures/sekil_boxplot_v14_v10_v4_time.png", bbox_inches="tight")
    plt.close()
    print("[INFO] Kutu grafikleri kaydedildi: figures/sekil_boxplot_v14_v10_v4_time.png")

if __name__ == "__main__":
    print("=" * 60)
    print("ADIM 1: KEŞİFSEL VERİ ANALİZİ (EDA) SÜRECİ BAŞLATILDI")
    print("=" * 60)
    
    df = veriyi_yukle()
    sekil1_sinif_dagilimi(df)
    sekil2_amount_dagilimi(df)
    sekil_boxplot_analizi(df)
    
    print("=" * 60)
    print("ADIM 1: İŞLEMLER BAŞARIYLA TAMAMLANDI")
    print("=" * 60)
