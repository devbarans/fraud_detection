# Python 3.10 slim tabanlı resmi imajı kullanıyoruz
FROM python:3.10-slim

# Çalışma dizinini /app olarak belirliyoruz
WORKDIR /app

# Derleme ve derleme bağımlılıkları için gerekli araçları yüklüyoruz
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pip önbelleğinden yararlanmak için öncelikle bağımlılıkları kopyalayıp kuruyoruz
COPY requirements.txt .

# Bağımlılıkları kuruyoruz (pip önbelleğini temiz tutarak imaj boyutunu küçültüyoruz)
RUN pip install --no-cache-dir -r requirements.txt

# Projedeki tüm kodları konteynere kopyalıyoruz
COPY . .

# pipeline scriptine çalıştırma yetkisi veriyoruz
RUN chmod +x run_pipeline.sh

# Varsayılan çalıştırma komutu olarak pipeline scriptimizi tanımlıyoruz
CMD ["./run_pipeline.sh"]
