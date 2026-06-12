#!/bin/bash
# Hatalarda calismayi durdur
set -e

echo "=========================================================="
echo "Kredi Kartı Sahtekarlığı Tespit Sistemi Pipeline Başlatılıyor"
echo "=========================================================="

# 1. EDA Adımı
echo ""
echo "[Adım 1/3] Keşifsel Veri Analizi (01_eda.py) yürütülüyor..."
python 01_eda.py

# 2. Ön İşleme ve SMOTE Adımı
echo ""
echo "[Adım 2/3] Veri Ön İşleme ve SMOTE (02_preprocess_smote.py) yürütülüyor..."
python 02_preprocess_smote.py

# 3. Model Eğitimi ve Karşılaştırma Adımı
echo ""
echo "[Adım 3/3] Model Eğitimi ve Performans Karşılaştırması (03_modeller.py) yürütülüyor..."
python 03_modeller.py

echo ""
echo "=========================================================="
echo "Pipeline Başarıyla Tamamlandı"
echo "Grafikler ve model çıktıları ilgili dizinlerde güncellendi."
echo "=========================================================="
