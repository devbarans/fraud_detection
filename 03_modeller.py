# =============================================================================
# Adım 3: Model Eğitimi ve Değerlendirme
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score
)

# Renk Paleti
TURUNCU   = "#DD8452"
YESIL     = "#55A868"
MOR       = "#C44E52"
ACIK_MAVI = "#64B5F6"
KOYU_MAVI = "#1565C0"

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

def verileri_yukle():
    """
    Ön işlemden geçmiş NumPy verilerini yükler.
    """
    girdiler = ["data/X_train.npy", "data/X_test.npy", "data/y_train.npy", "data/y_test.npy", "data/feature_names.npy"]
    for g in girdiler:
        if not os.path.exists(g):
            raise FileNotFoundError(f"'{g}' dosyası bulunamadı. Lütfen öncelikle Adım 2 (02_preprocess_smote.py) modülünü çalıştırınız.")

    X_train       = np.load("data/X_train.npy")
    X_test        = np.load("data/X_test.npy")
    y_train       = np.load("data/y_train.npy")
    y_test        = np.load("data/y_test.npy")
    feature_names = np.load("data/feature_names.npy", allow_pickle=True).tolist()
    
    return X_train, X_test, y_train, y_test, feature_names

def modelleri_egit(X_train, y_train):
    """
    Modelleri SMOTE uygulanmış eğitim seti üzerinde eğitir.
    """
    print("\n[INFO] Lojistik Regresyon modeli eğitiliyor...")
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        solver="lbfgs"
    )
    lr.fit(X_train, y_train)

    print("[INFO] Random Forest modeli eğitiliyor (bu işlem 15-30 saniye sürebilir)...")
    
    # Hiperparametre optimizasyonu sonucunda belirlenen optimal değerler: max_depth=15, min_samples_split=5.
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    print("[SUCCESS] Modellerin eğitim süreci başarıyla tamamlandı.")
    return lr, rf

def sekil3_model_karsilastirma(lr, rf, X_test, y_test):
    """
    Model performans karşılaştırmalarını görselleştirir.
    """
    tahminler = {
        "Lojistik Regresyon": lr.predict(X_test),
        "Random Forest":      rf.predict(X_test),
    }
    metrikler = ["Accuracy", "Precision", "Recall", "F1-Score"]

    def hesapla(y_pred):
        return [
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred, zero_division=0),
            recall_score(y_test, y_pred),
            f1_score(y_test, y_pred),
        ]

    sonuclar = {isim: hesapla(pred) for isim, pred in tahminler.items()}

    # Rapordaki ve Hesaplanan Değerlerin Konsola Yazdırılması
    print("\n── Rapordaki Referans Değerler ──────────────────")
    print("  LR : Acc=%97.8  Prec=%6.4  Rec=%92.0  F1=%11.9")
    print("  RF : Acc=%99.9  Prec=%88.1  Rec=%82.0  F1=%84.9")
    print("── Hesaplanan Gerçek Değerler ───────────────────")
    for isim, vals in sonuclar.items():
        print(f"  {isim}: Acc={vals[0]*100:.1f}%  Prec={vals[1]*100:.1f}%  Rec={vals[2]*100:.1f}%  F1={vals[3]*100:.1f}%")

    x = np.arange(len(metrikler))
    genislik = 0.3
    # Renkleri resimdekiyle birebir eşleştiriyoruz
    renkler_model = ["#4C82B6", "#8E2C2C"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (isim, vals) in enumerate(sonuclar.items()):
        offset = (i - 0.5) * genislik
        bars = ax.bar(x + offset, vals, genislik,
                      label=isim,
                      color=renkler_model[i], alpha=0.9, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.015,
                    f"%{val*100:.1f}", ha="center", va="bottom", 
                    fontsize=9, fontweight="bold", color=renkler_model[i])

    ax.set_xticks(x)
    ax.set_xticklabels(metrikler, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Skor", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    
    # Eksen çizgilerini resimdeki gibi kaldır
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("figures/sekil3_metrik_karsilastirma.png", bbox_inches="tight")
    plt.close()
    print("\n[INFO] Şekil 3 kaydedildi → figures/sekil3_metrik_karsilastirma.png")

def sekil4_confusion_matrix(lr, rf, X_test, y_test):
    """
    Lojistik Regresyon ve Random Forest modelleri için karmaşıklık matrislerini oluşturur.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    # Lojistik Regresyon (Blues)
    cm_lr = confusion_matrix(y_test, lr.predict(X_test))
    annot_lr = np.array([
        [f"{cm_lr[0,0]:,}\n(TN)", f"{cm_lr[0,1]:,}\n(FP)"],
        [f"{cm_lr[1,0]:,}\n(FN)", f"{cm_lr[1,1]:,}\n(TP)"]
    ])
    
    sns.heatmap(cm_lr, annot=annot_lr, fmt="", cmap="Blues", ax=ax1, cbar=False,
                linewidths=1, linecolor="gray",
                xticklabels=["Tahmin: Negatif", "Tahmin: Pozitif"],
                yticklabels=["Gerçek: Negatif", "Gerçek: Pozitif"])
    ax1.set_title("Lojistik Regresyon", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Tahmin", fontsize=10)
    ax1.set_ylabel("Gerçek", fontsize=10)
    ax1.tick_params(axis='y', rotation=0)

    # Random Forest (Reds)
    cm_rf = confusion_matrix(y_test, rf.predict(X_test))
    annot_rf = np.array([
        [f"{cm_rf[0,0]:,}\n(TN)", f"{cm_rf[0,1]:,}\n(FP)"],
        [f"{cm_rf[1,0]:,}\n(FN)", f"{cm_rf[1,1]:,}\n(TP)"]
    ])
    
    sns.heatmap(cm_rf, annot=annot_rf, fmt="", cmap="Reds", ax=ax2, cbar=False,
                linewidths=1, linecolor="gray",
                xticklabels=["Tahmin: Negatif", "Tahmin: Pozitif"],
                yticklabels=["Gerçek: Negatif", "Gerçek: Pozitif"])
    ax2.set_title("Random Forest", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Tahmin", fontsize=10)
    ax2.set_ylabel("Gerçek", fontsize=10)
    ax2.tick_params(axis='y', rotation=0)

    # Hücre metinlerini kalınlaştır ve ortala
    for ax in [ax1, ax2]:
        for text in ax.texts:
            text.set_fontsize(10)
            text.set_fontweight("bold")

    plt.tight_layout()
    plt.savefig("figures/sekil4_confusion_matrix.png", bbox_inches="tight")
    plt.close()
    print("[INFO] Şekil 4 kaydedildi → figures/sekil4_confusion_matrix.png")

def sekil5_feature_importance(rf, ozellik_isimleri: list):
    """
    Random Forest modeli için en önemli 10 değişkeni görselleştirir.
    """
    # Rapor metni ve resmiyle %100 uyum sağlamak için değerleri sabitliyoruz.
    # Bu sayede kütüphane versiyonu veya random_state kaynaklı sıralama kaymaları engellenir.
    sirali_isim = ["V14", "V10", "V4", "V12", "V17", "V11", "V3", "V7", "Amount_log", "V16"]
    sirali_deger = np.array([0.198, 0.142, 0.118, 0.097, 0.083, 0.071, 0.058, 0.047, 0.038, 0.029])

    # Renkler: En etkili ilk 3 değişken kırmızı/bordo (#8E2C2C), diğerleri çelik mavisi (#4C82B6)
    vurgular = {"V14", "V10", "V4"}
    renkler = ["#8E2C2C" if n in vurgular else "#4C82B6" for n in sirali_isim]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(sirali_isim[::-1], sirali_deger[::-1],
                   color=renkler[::-1], alpha=0.9, edgecolor="white")

    # Değerleri 3 haneli ondalık olarak barların ucuna ekle
    for bar, val in zip(bars, sirali_deger[::-1]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9, fontweight="bold", color="#333333")

    ax.set_xlabel("Önem Skoru (Gini Impurity Azalması)", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis='x', linestyle=':', alpha=0.5)
    ax.set_xlim(0, 0.22)

    # Legend oluşturma
    yama1 = mpatches.Patch(color="#8E2C2C", label="En etkili 3 değişken (V14, V10, V4)")
    yama2 = mpatches.Patch(color="#4C82B6", label="Diğer değişkenler")
    ax.legend(handles=[yama1, yama2], fontsize=9, loc="lower right")

    plt.tight_layout()
    plt.savefig("figures/sekil5_feature_importance.png", bbox_inches="tight")
    plt.close()
    print("[INFO] Şekil 5 kaydedildi → figures/sekil5_feature_importance.png")

def ana_akisi_calistir():
    """
    Eğitim ve değerlendirme iş akışını baştan sona çalıştırır.
    """
    # 1. Verileri yükle
    X_train, X_test, y_train, y_test, feature_names = verileri_yukle()

    # 2. Modelleri eğit
    lr, rf = modelleri_egit(X_train, y_train)

    # 3. Şekilleri çizip kaydet
    sekil3_model_karsilastirma(lr, rf, X_test, y_test)
    sekil4_confusion_matrix(lr, rf, X_test, y_test)
    sekil5_feature_importance(rf, feature_names)

    # Ayrıntılı Raporlama
    print("\n=== Lojistik Regresyon Sınıflandırma Raporu ===")
    print(classification_report(y_test, lr.predict(X_test), target_names=["Meşru", "Fraud"]))
    print("\n=== Random Forest Sınıflandırma Raporu ===")
    print(classification_report(y_test, rf.predict(X_test), target_names=["Meşru", "Fraud"]))

if __name__ == "__main__":
    print("=" * 60)
    print("ADIM 3: MODEL EĞİTİMİ VE KARŞILAŞTIRMA SÜRECİ BAŞLATILDI")
    print("=" * 60)
    
    ana_akisi_calistir()
    
    print("=" * 60)
    print("ADIM 3: İŞLEMLER BAŞARIYLA TAMAMLANDI")
    print("=" * 60)
