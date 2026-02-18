# SilentTalk-3D
## Kamera Tabanlı Sessiz İletişim Tanıma Sistemi (3D Derin Öğrenme)

SilentTalk-3D, konuşma yetisini kullanamayan bireylerin ses çıkarmadan iletişim kurabilmesini sağlayan yapay zekâ tabanlı görsel iletişim tanıma sistemidir.

Sistem kameradan alınan yüz hareketlerini analiz eder ve kullanıcının iletmek istediği iletişim kategorisini gerçek zamanlı olarak ekranda gösterir.

Bu proje özellikle:

- hastane ortamları
- rehabilitasyon süreçleri
- eller serbest bilgisayar kontrolü

gibi senaryolar için geliştirilmiş bir prototiptir.

---

## Sistem Nasıl Çalışır

<img width="1858" height="762" alt="Ekran görüntüsü 2026-02-18 214826" src="https://github.com/user-attachments/assets/31073a51-5177-48c0-962e-014bf308c4ba" />


Sistem tek bir fotoğrafı değil, zaman içindeki hareketi analiz eder.

1. Kamera görüntüsü alınır
2. Yüz bölgesi tespit edilir (MediaPipe)
3. 16 framelik kısa bir video klibi oluşturulur
4. 3D CNN modeli hareketi analiz eder
5. Tahmin edilen sınıf ekranda gösterilir

Yani model görüntüyü değil **hareket paternini öğrenir**.

<img width="880" height="654" alt="Ekran görüntüsü 2026-02-18 212228" src="https://github.com/user-attachments/assets/d1841c7b-3669-4ca0-8f1c-561f7b6de5fd" />
<img width="885" height="598" alt="Ekran görüntüsü 2026-02-18 212305" src="https://github.com/user-attachments/assets/c5418c97-e154-459c-8bb5-409a9b7bcf57" />
<img width="889" height="613" alt="Ekran görüntüsü 2026-02-18 212324" src="https://github.com/user-attachments/assets/a623b66d-e243-4809-8a41-76da26007490" />
<img width="878" height="666" alt="Ekran görüntüsü 2026-02-18 212404" src="https://github.com/user-attachments/assets/053ddf85-4cb4-4c9c-b1f2-7251af644068" />






---

## Kullanılan Model

**R(2+1)D-18 — 3D Convolutional Neural Network**

Bu mimari video anlayabilen bir yapay zekâ modelidir.  
Ardışık kareler arasındaki yüz hareketlerini öğrenerek sınıflandırma yapar.

---

## Proje Yapısı

SilentTalk-3D/
│
├── src/
│ ├── preprocess/ → video → klip üretme
│ ├── train/ → model eğitimi
│ └── app/ → canlı kamera tahmin
│
├── models/ → eğitilmiş model
├── data_raw/ → paylaşılmadı
├── data_processed/ → paylaşılmadı
│
├── requirements.txt
└── README.md


---

## Veri Seti

Bu projede kullanılan veri seti tarafımızca kamera ile toplanmıştır.

Özellikler:

- 1–3 saniyelik kısa yüz videoları
- Her video belirli bir iletişim amacını temsil eder
- 16 framelik kliplere dönüştürülerek eğitilmiştir
- Zaman tabanlı yüz hareketi içerir

> Veri seti kişisel görüntü içerdiği için paylaşılmamaktadır.


---

## Model Eğitimi

### 1) Klip üretme

python src/preprocess/make_clips.py

## 2) Model eğitme
python src/train/train_3dcnn.py



## Canlı Tahmin Uygulaması
streamlit run src/app/live_app.py


Kamera açılır ve sistem gerçek zamanlı tahmin yapar.

Arayüz

## Kullanılan Teknolojiler

* Python

* PyTorch

* 3D CNN

* OpenCV

* MediaPipe

* Streamlit

# Mevcut Durum

Sistem kısa yüz hareketlerini gerçek zamanlı analiz edip stabil sonuç üretebilmektedir.

# Geliştirilebilir Yönler

* Farklı kişilerden veri ekleme

* Daha gelişmiş video modelleri

* Mobil uygulama

* Ses üretimi entegrasyonu

# Amaç

Bu proje,  hareket analizi kullanarak sessiz iletişim kurulabileceğini gösteren yardımcı bir teknoloji prototipidir.
