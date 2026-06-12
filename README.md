# Kredi Kartı Dolandırıcılığı Tespiti (Credit Card Fraud Detection)

[![Python Version](https://img.shields.io/badge/Python-3.10-blue.svg?style=flat-square&logo=python)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue.svg?style=flat-square&logo=docker)](https://www.docker.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.7.2-orange.svg?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-0.14.1-red.svg?style=flat-square)](https://imbalanced-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

Bu çalışma, **Eskişehir Osmangazi Üniversitesi, İstatistik Bölümü Veri Madenciliği Dersi Dönem Sonu Projesi** kapsamında geliştirilmiştir. Araştırma ve uygulama süreci, orijinal Kaggle **Credit Card Fraud Detection** veri seti üzerinde makine öğrenmesi algoritmaları (Lojistik Regresyon ve Random Forest) kullanılarak dolandırıcılık (fraud) işlemlerini yüksek hassasiyetle tespit etmeyi amaçlar.

Sistem, hem yerel ortamda hem de izole edilmiş bir **Docker** konteyneri içerisinde tamamen tekrarlanabilir (reproducible) bir şekilde çalışacak şekilde tasarlanmıştır.

---

## Proje Özellikleri ve Teknolojik Altyapı

- **Docker Entegrasyonu:** Sürüm çakışmalarını ortadan kaldıran, tek bir komutla kurulup çalıştırılabilen izole Docker mimarisi.
- **Otomatik Kaggle Veri Temini (`01_eda.py`):** Proje dizininde `creditcard.csv` dosyası bulunmadığı durumlarda, `kagglehub` kütüphanesi aracılığıyla **orijinal 150 MB'lık Kaggle veri setini** anonim olarak (API anahtarı veya kimlik bilgisine ihtiyaç duymadan) otomatik olarak indirir ve `data/` dizinine yerleştirir.
- **Dizin Eşleme ve Çıktı Yönetimi (Volume Mounts):** Docker konteyneri içerisinde üretilen tüm grafik analiz çıktıları (`.png`) ve ön işlenmiş veri dosyaları (`.npy`), anında ana bilgisayarın çalışma dizinine senkronize edilir.
- **Veri Sızıntısını Önleyen SMOTE Tasarımı:** Eğitim ve test setleri ayrıldıktan sonra SMOTE algoritmasının sadece eğitim kümesine uygulanması sağlanarak literatürde sıkça yapılan *Veri Sızıntısı (Data Leakage)* hatası engellenmiştir.

---

## Proje Dizin Yapısı

```
fraud_detection/
├── 01_eda.py                 # Adım 1 — Keşifsel Veri Analizi, Otomatik Veri İndirme ve Grafik 1-2 Çizimleri
├── 02_preprocess_smote.py    # Adım 2 — Veri Bölümleme, Ölçeklendirme ve SMOTE Uygulaması
├── 03_modeller.py            # Adım 3 — Modellerin Eğitimi, Karşılaştırma Raporu ve Grafik 3-4-5 Çizimleri
├── run_pipeline.sh           # Konteyner/Lokal ortamda uçtan uca tüm hattı çalıştıran otomasyon betiği
├── Dockerfile                # Docker imaj yapılandırma dosyası (İzole ve tekrarlanabilir çalışma)
├── requirements.txt          # Python kütüphane bağımlılıkları listesi
├── LICENSE                   # MIT Lisans belgesi
└── README.md                 # Proje detaylı dokümantasyonu (Bu dosya)
```

---

## Kurulum ve Çalıştırma Kılavuzu

### Yöntem 1: Docker ile Çalıştırma (Önerilen)

Docker, projenin sisteminizdeki mevcut Python veya kütüphane sürümlerinden etkilenmeden izole ve temiz bir şekilde çalışmasını sağlar.

#### 1. Docker İmajını İnşa Edin:
Terminalde proje ana dizinindeyken aşağıdaki komutu çalıştırın:
```bash
docker build -t fraud-detection-app .
```

#### 2. Konteyneri Çalıştırın:
İndirilen veri setinin bilgisayarınızda önbelleğe alınması ve konteyner içinde üretilen tüm çıktı grafiklerinin yerel dizininize anında aktarılması için aşağıdaki dizin eşleme (volume mounts) komutunu kullanın:
```bash
docker run --rm \
  -v "$(pwd)":/app \
  -v ~/.cache/kagglehub:/root/.cache/kagglehub \
  fraud-detection-app
```
*(Bu işlem başlatıldığında, `creditcard.csv` dosyası otomatik olarak indirilecek, veri ön işleme, SMOTE ve model eğitimi adımları uçtan uca çalıştırılacak ve sonuç grafikleri yerel dizininizdeki `figures/` klasöründe güncellenecektir.)*

---

### Yöntem 2: Yerel Bilgisayarda Çalıştırma (Python)

Docker kullanmadan doğrudan Python ortamınızda çalıştırmak için aşağıdaki adımları takip edebilirsiniz:

#### 1. Bağımlılıkları Yükleyin:
```bash
pip install -r requirements.txt
```

#### 2. Pipeline Otomasyonunu Çalıştırın:
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

---

## Veri Seti Dağılımı ve Ön İşleme İstatistikleri

Orijinal veri setinin analizi ve eğitim/test ayrımı sonucunda elde edilen dağılımlar aşağıda özetlenmiştir:

- **Toplam İşlem Sayısı:** 284.807 adet gözlem
- **Sınıf Dağılımı:** 284.315 Meşru İşlem (%99,827) | 492 Dolandırıcılık İşlemi (%0,173)
- **Eğitim Seti (Stratified Train):** 227.845 gözlem (394 adet Dolandırıcılık)
- **Test Seti (Stratified Test):** 56.962 gözlem (98 adet Dolandırıcılık)
- **SMOTE Sonrası Dengelenmiş Eğitim Seti:** 227.451 Meşru + 227.451 Dolandırıcılık (Toplam: 454.902 gözlem)

---

## Model Performansı ve Değerlendirme Sonuçları

Modeller, SMOTE ile dengelenmiş eğitim kümesinde eğitildikten sonra, **orijinal dengesiz yapısı korunan %20'lik test seti (56.962 gözlem)** üzerinde test edilmiştir.

### Tablo 1: Model Karşılaştırma Sonuçları

| Değerlendirme Metriği | Lojistik Regresyon | Random Forest (Rastgele Orman) |
| :--- | :---: | :---: |
| **Accuracy (Doğruluk)** | %97,3 | **%99,9** |
| **Precision (Hassasiyet)** | %5,6 | **%87,0** |
| **Recall (Duyarlılık)** | **%91,8** | %81,6 |
| **F1-Score** | %10,5 | **%84,2** |

### Metrik Analizleri ve Model Değerlendirmesi

1. **Random Forest Performansı:** Random Forest modeli, **%99,9 doğruluk** ve **%84,2 F1-Score** ile dengesiz veri kümesinde üstün bir sınıflandırma performansı sergilemiştir. Dolandırıcılık olarak tahmin ettiği işlemlerin **%87,0'si gerçekten dolandırıcılıktır (Precision)** ve dolandırıcılık işlemlerinin **%81,6'sını başarılı bir şekilde tespit etmiştir (Recall)**.
2. **Lojistik Regresyon Karar Mekanizması:** Lojistik Regresyon modeli **%91,8 Duyarlılık (Recall)** ile dolandırıcılık işlemlerinin büyük bir kısmını yakalamış olsa da, **%5,6 gibi düşük bir Hassasiyet (Precision)** göstermiştir. Bu durum, modelin çok sayıda meşru işleme hatalı şekilde dolandırıcılık teşhisi koyduğunu (False Positive) ve operasyonel maliyet oluşturma riski taşıdığını gösterir.

---

## Analiz Çıktıları ve Görselleştirmeler

Pipeline başarıyla tamamlandığında, projenin `figures/` dizininde aşağıdaki analiz dosyaları otomatik olarak oluşturulur ve güncellenir:

| Dosya Adı | Açıklama |
| :--- | :--- |
| `sekil1_sinif_dagilimi.png` | Veri setindeki sınıf dengesizliğini oransal ve logaritmik ölçekte gösteren pasta ve sütun grafikleridir. |
| `sekil2_amount_dagilimi.png` | İşlem tutarı (`Amount`) dağılımındaki yüksek çarpıklığı (Skewness: 16.98) logaritmik dönüşümle (Skewness: 0.16) nasıl simetrik hale getirdiğimizi gösterir. |
| `sekil_boxplot_v14_v10_v4_time.png` | Dolandırıcılık tespiti için ayırt edici olan önemli PCA bileşenlerinin (V14, V10, V4) ve Time bileşeninin sınıflar arasındaki kutu grafiklerini sunar. |
| `sekil3_metrik_karsilastirma.png` | Eğitilen modellerin Accuracy, Precision, Recall ve F1-Score metriklerini test kümesi üzerinden görsel olarak kıyaslar. |
| `sekil4_confusion_matrix.png` | Modellerin Doğru Negatif (TN), Yanlış Pozitif (FP), Yanlış Negatif (FN) ve Doğru Pozitif (TP) tahmin sayılarını gösteren karmaşıklık matrisidir. |
| `sekil5_feature_importance.png` | Random Forest modeline göre dolandırıcılık tespitinde en yüksek ayırt ediciliğe sahip ilk 10 değişkenin (örn. `V14`, `V10`, `V4`) önem sıralamasını listeler. |
| `sekil6_smote_gorseli.png` | SMOTE öncesi ve sonrası sınıf dengelenmesini sayısal olarak ve en önemli iki bileşen (`V14` ve `V10`) arasındaki geometrik dağılımı 2D uzayda karşılaştırmalı olarak sunar. |

---

## Proje Metodolojisi ve Tasarım Kararları

### 1. SMOTE Algoritmasının Sadece Eğitim Setine Uygulanması (Veri Sızıntısı & Kaufman vd. 2012)
Modelleme sürecinde veri sızıntısının (data leakage) önüne geçebilmek adına SMOTE algoritması yalnızca eğitim setine (train set) uygulanmış, test setine kesinlikle müdahale edilmemiştir. Kaufman ve arkadaşlarının (2012) tanımladığı üzere, eğer bu işlem train-test ayrımından önce tüm veri setine uygulansaydı, üretilen yapay gözlemler hem eğitim hem de test tarafındaki gerçek örnekleri baz alarak sentezlenecekti. Bu durum, modelin test verisinin geometrik yapısını ve dağılımını dolaylı yoldan henüz eğitim aşamasındayken öğrenmesine (bilgi sızıntısına) yol açardı. Kaufman vd. (2012) bu konuyu detaylıca incelemiş ve özellikle over-sampling sırasındaki sızıntıyı, gerçekçi olmayan yüksek performans tahminlerine ve dolayısıyla yanıltıcı metrik şişmelerine (overoptimism) yol açan en yaygın metodolojik hatalardan biri olarak işaret etmiştir.

Bu çalışmada, SMOTE sadece `X_train` üzerinde çalıştırılarak test kümesi tamamen izole tutulmuş ve model performansı gerçekçi sınırlarında doğrulanmıştır.

### 2. Dengesiz Veri Kümelerinde Doğruluk (Accuracy) Metriğinin Yetersizliği
Kredi kartı veri setlerinin %99,83'ü normal işlemlerden oluşmaktadır. Hiçbir analiz yapmadan tüm işlemleri "Normal" olarak tahmin eden basit bir sınıflandırıcı bile **%99,83 Accuracy** değerine ulaşacaktır. Dolayısıyla bu tarz dengesiz problemlerde temel alınması gereken metrikler **Recall (Duyarlılık)**, **Precision (Hassasiyet)** ve ikisinin dengeli birleşimi olan **F1-Score**'dur.

### 3. Hiperparametre Optimizasyonu ve Hesaplama/Zaman Kısıtları
SMOTE uygulamasıyla birlikte eğitim setindeki gözlem sayısı ve veri hacmi ciddi ölçüde büyümüştür (toplam 454.902 gözlem). Bu genişletilmiş veri kümesi üzerinde Random Forest algoritması çalıştırıldığında, ağaçların dal yapısı ve derinliği ciddi ölçüde kabardığı için işlem süresi belirgin şekilde uzamıştır. İlk planlamada GridSearchCV yöntemi kullanılarak hem `n_estimators` hem de `max_depth` parametreleri üzerinde kapsamlı ve sistematik bir hiperparametre taraması yapılması hedeflenmiştir. 

Ancak ön denemelerdeki yüksek zaman maliyetleri ve pratik kısıtlar nedeniyle bu kapsam daraltılmak zorunda kalınmıştır. İşlem süresini makul bir sınırda tutabilmek adına `n_estimators` parametresi sabit bir değerde (100) tutulmuş; modelin aşırı öğrenmesini (overfitting) ve daldan dallanma karmaşıklığını kontrol etmek amacıyla tarama yalnızca `max_depth` ve `min_samples_split` gibi kritik yapısal hiperparametreler üzerinden yürütülmüştür. Tarama sonucunda optimum model derinliği `max_depth=15`, dallanma kısıtı `min_samples_split=5` olarak belirlenmiştir.

---

---

## Lisans

Bu proje **MIT Lisansı** altında lisanslanmıştır. Detaylı bilgi için `LICENSE` dosyasını inceleyebilirsiniz.
