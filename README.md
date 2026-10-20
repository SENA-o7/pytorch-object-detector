# PyTorch ile Sıfırdan Nesne Tespiti (Object Detector)

Bu proje, Google Colab üzerinde bir görüntüdeki nesnenin **ne olduğunu** ve **nerede olduğunu** aynı anda tahmin eden bir model eğitir. Kodlar, PyImageSearch'ün [*Training an object detector from scratch in PyTorch*](https://pyimagesearch.com/2021/11/01/training-an-object-detector-from-scratch-in-pytorch/) (01.11.2021) yazısını temel alır. Güncel Colab ortamında çıkan sürüm hataları giderilmiştir.


## 1. Ödev maddeleri ve karşılıkları

| No | Ödev maddesi | Notebook'taki yeri |
|---|---|---|
| 1 | Object detector eğitimi ve model dosyası üretimi | Bölüm 4–7, `detector.pth` |
| 2 | Veri setinin Kaggle API ile Colab'a çekilmesi | Bölüm 2 |
| 3 | Sürüm kaynaklı hataların giderilmesi | Koddaki `# [SÜRÜM DÜZELTMESİ]` yorumları ve Bölüm 9 |
| 4 | Accuracy ve Loss grafikleri | Bölüm 8 |
| 5 | Grafik arayüzsüz inference fonksiyonu | Bölüm 10, `tahmin_et()` |

Projedeki dosyalar:

| Dosya | Açıklama |
|---|---|
| `object_detector_colab.ipynb` | Ana notebook. ResNet50 kullanır, GPU ile çalıştırılması önerilir. |
| `Nesne_tespiti/Nesne_Tespiti_Egitimi.ipynb` | Alternatif sürüm. ResNet18 kullanır, CPU'da daha hızlı çalışır. |
| `README.md` | Bu belge. |

---

## 2. Temel kavramlar

Koda geçmeden önce projede geçen kavramları kısaca açıklayalım.

### 2.1 Sınıflandırma, yerelleştirme ve nesne tespiti

| Görev | Sorduğu soru | Çıktı |
|---|---|---|
| Sınıflandırma (classification) | Görüntüde ne var? | Bir sınıf adı, ör. `airplanes` |
| Yerelleştirme (localization) | Nesne nerede? | Bir kutu, ör. `(x1, y1, x2, y2)` |
| Nesne tespiti (object detection) | Ne var ve nerede? | Sınıf adı + kutu |

Bu projedeki her görüntüde **tek bir nesne** vardır. Yani yaptığımız iş tam olarak "sınıflandırma + yerelleştirme"dir. YOLO veya Faster R-CNN gibi gelişmiş modeller aynı görüntüde birden fazla nesne bulabilir. Bizim modelimiz ise nesne tespitinin en temel hâlini öğretmek için tasarlanmıştır.

### 2.2 Bounding box (sınırlayıcı kutu)

Bir nesnenin yeri, onu çevreleyen dikdörtgenin iki köşesiyle ifade edilir:

```
(x1, y1) +-------------------+
         |                   |
         |      nesne        |
         |                   |
         +-------------------+ (x2, y2)
```

- `(x1, y1)`: sol üst köşe
- `(x2, y2)`: sağ alt köşe

Görüntülerde koordinat sistemi sol üst köşeden başlar. `x` sağa doğru, `y` **aşağıya** doğru artar.

### 2.3 Transfer learning (aktarımlı öğrenme)

Bir modeli sıfırdan eğitmek milyonlarca görüntü ve uzun süre gerektirir. Bunun yerine, **ImageNet** (1,2 milyon görüntü, 1000 sınıf) üzerinde önceden eğitilmiş bir model alırız. Bu model kenar, doku ve şekil gibi genel görsel özellikleri zaten tanımayı öğrenmiştir.

Biz bu modelin gövdesini (backbone) **dondururuz**, yani ağırlıklarını değiştirmeyiz. Üzerine kendi görevimiz için yeni katmanlar ekler ve yalnızca bunları eğitiriz. Böylece:
- az veriyle iyi sonuç alırız,
- eğitim çok daha hızlı biter,
- model ezberlemeye (overfitting) daha az eğilimli olur.

Bu projede gövde olarak **ResNet50** kullanılır. ResNet50 bir görüntüyü 2048 sayıdan oluşan bir **özellik vektörüne** dönüştürür. Bu vektör, görüntünün sayısal bir özeti gibi düşünülebilir.

### 2.4 Çok görevli öğrenme (multi-task learning)

Modelimiz aynı özellik vektöründen iki farklı tahmin üretir. Buna çok görevli öğrenme denir:

```
                                    +--> Sınıflandırıcı kafa --> 3 skor (her sınıf için bir tane)
Görüntü --> ResNet50 gövde --> 2048 |
                                    +--> Kutu regresör kafası --> 4 sayı (x1, y1, x2, y2)
```

İki görev farklı türde olduğu için her biri farklı bir kayıp fonksiyonu kullanır.

### 2.5 Kayıp (loss) fonksiyonları

Kayıp fonksiyonu, modelin tahmininin gerçek değerden ne kadar uzak olduğunu tek bir sayıyla ölçer. Eğitimin amacı bu sayıyı küçültmektir.

**Sınıflandırma için Cross-Entropy Loss:**
Model her sınıf için bir skor (logit) üretir. Bu skorlar **Softmax** ile toplamı 1 olan olasılıklara çevrilir:

```
Skorlar (logit):   airplanes: 4.1   motorbikes: 0.3   schooners: -1.2
Softmax sonrası:   airplanes: 0.97  motorbikes: 0.02  schooners: 0.01
```

Cross-Entropy, doğru sınıfa verilen olasılık 1'e ne kadar yakınsa o kadar küçük bir değer verir. Model yanlış sınıfa emin bir şekilde yüksek olasılık verirse kayıp büyür.

**Kutu için MSE (Mean Squared Error, ortalama kare hata):**
Tahmin edilen 4 koordinat ile gerçek koordinatlar arasındaki farkların karelerinin ortalamasıdır:

```
MSE = [(x1_tahmin - x1_gercek)² + (y1_tahmin - y1_gercek)² + (x2_tahmin - x2_gercek)² + (y2_tahmin - y2_gercek)²] / 4
```

**Toplam kayıp:**

```
toplam_kayip = LABELS * siniflandirma_kaybi + BBOX * kutu_kaybi
```

`LABELS` ve `BBOX` ağırlıklarıdır (bu projede ikisi de 1.0). Birini büyütmek, modelin o göreve daha fazla önem vermesini sağlar.

### 2.6 Epoch, batch ve öğrenme oranı

- **Batch:** Model ağırlıklarını her adımda tüm veriye bakarak değil, küçük bir grup görüntüye (ör. 32 tane) bakarak günceller. Bu gruba batch denir.
- **Epoch:** Modelin eğitim verisinin tamamını bir kez görmesidir. 20 epoch, verinin 20 kez baştan sona dolaşılması demektir.
- **Öğrenme oranı (learning rate):** Her adımda ağırlıkların ne kadar değiştirileceğini belirler. Çok büyükse eğitim kararsız olur, çok küçükse eğitim çok yavaş ilerler.

### 2.7 Accuracy ve overfitting

- **Accuracy (doğruluk):** Doğru sınıflandırılan görüntülerin oranı. 0.95, görüntülerin %95'inin doğru tahmin edildiği anlamına gelir.
- **Overfitting (aşırı öğrenme / ezberleme):** Model eğitim verisini ezberler ve yeni görüntülerde başarısız olur. Grafikte eğitim başarısı artmaya devam ederken test başarısının durması veya düşmesiyle anlaşılır.

---

## 3. Veri seti

- **Kaynak:** [Caltech101 | Airplanes, Motorbikes & Schooners (Kaggle)](https://www.kaggle.com/datasets/maricinnamon/caltech101-airplanes-motorbikes-schooners)
- **Kökeni:** Caltech-101 veri setinden seçilmiş 3 sınıflı hafif bir alt küme
- **Sınıflar:**

| Sınıf | Türkçe |
|---|---|
| `airplanes` | Uçaklar |
| `motorbikes` | Motosikletler |
| `schooners` | Yelkenli gemiler |

Veri seti iki parçadan oluşur:

1. **Görüntüler:** Her sınıf için ayrı klasörde JPEG dosyaları. Görüntülerin boyutları birbirinden farklıdır.
2. **Etiketler (annotation):** Her görüntü için nesneyi çevreleyen kutunun koordinatları. CSV dosyalarında satır yapısı şöyledir:

```
filename, start_x, start_y, end_x, end_y, label
```

Caltech-101'in orijinal sürümünde etiketler MATLAB'ın `.mat` formatındadır ve koordinat sırası farklıdır (`ymin, ymax, xmin, xmax`). Notebook her iki formatı da otomatik olarak tanır, koordinatları doğru sıraya çevirir.

**Not:** Sınıflar dengeli değildir, yani her sınıftaki görüntü sayısı farklıdır. Bu yüzden veri bölünürken `stratify` kullanılır (bkz. Adım 5).

---

## 4. Projenin adım adım işleyişi

Notebook'taki bölümler aşağıdaki sırayla çalışır.

### Adım 0 – Veri seti incelemesi
Veri setinin yapısı ve modelin genel mantığı açıklanır. Kod içermez.

### Adım 1 – Ortam kontrolü
PyTorch, torchvision, OpenCV gibi kütüphanelerin sürümleri yazdırılır. Bir hata çıktığında hangi sürümle çalışıldığını bilmek, hatanın nedenini bulmayı kolaylaştırır.

### Adım 2 – Kaggle API ile veriyi indirme
Kaggle, veri setlerini komut satırından indirmeye izin veren bir API sunar. Bu API'yi kullanmak için kimlik doğrulama gerekir:

1. Kaggle hesabından alınan `kaggle.json` dosyası Colab'a yüklenir.
2. Dosya, Kaggle'ın aradığı yer olan `~/.kaggle/kaggle.json` konumuna taşınır.
3. Dosya izinleri `600` yapılır (yalnızca sahibi okuyabilir). Yapılmazsa Kaggle güvenlik uyarısı verir.
4. `kaggle datasets download` komutuyla veri `/content/data` klasörüne indirilir ve açılır.

### Adım 3 – Ayarlar
Orijinal projede `config.py` dosyasında bulunan tüm ayarlar tek hücrede toplanır:

| Ayar | Değer | Açıklama |
|---|---|---|
| `GORUNTU_BOYUTU` | 224 | ResNet50'nin beklediği giriş boyutu |
| `INIT_LR` | 1e-4 | Öğrenme oranı |
| `NUM_EPOCHS` | 20 | Eğitim turu sayısı |
| `BATCH_SIZE` | 32 | Her adımda işlenen görüntü sayısı |
| `TEST_ORANI` | 0.20 | Verinin %20'si teste ayrılır |
| `LABELS`, `BBOX` | 1.0, 1.0 | Kayıp ağırlıkları |
| `NUM_WORKERS` | 2 | Veriyi paralel yükleyen işlem sayısı |
| `DEVICE` | `cuda` veya `cpu` | GPU varsa otomatik olarak kullanılır |

### Adım 4 – Etiketleri okuma
Her görüntü dosyası, etiket dosyasındaki kutu koordinatlarıyla eşleştirilir. Sonuç olarak her görüntü için şu bilgi elde edilir: `dosya yolu -> (sınıf, x1, y1, x2, y2)`.

Okumanın doğru yapıldığını görmek için her sınıftan bir örnek, kutusu çizilerek gösterilir. Bu adım atlanmamalıdır: koordinat sırası yanlış okunursa kutular nesnenin dışına düşer ve model yanlış şeyi öğrenir.

### Adım 5 – Veriyi hazırlama
Her görüntü için sırasıyla:

1. **Kutu normalizasyonu:** Koordinatlar görüntünün genişlik ve yüksekliğine bölünerek 0–1 arasına çekilir.
   ```python
   kutu = (x1 / w, y1 / h, x2 / w, y2 / h)
   ```
   *Neden?* Görüntüler farklı boyutlarda ve hepsi 224×224'e küçültülecek. Koordinatı orana çevirirsek boyut ne olursa olsun kutu aynı göreli yeri gösterir. Örneğin genişliği 400 piksel olan bir görüntüde `x1 = 100` ise oran 0.25'tir. Görüntü 224 piksele küçültülünce kutu `0.25 * 224 = 56` pikselden başlar.
2. **Renk dönüşümü:** OpenCV görüntüleri BGR sırasıyla okur. ResNet RGB sırasıyla eğitildiği için `BGR -> RGB` dönüşümü yapılır.
3. **Yeniden boyutlandırma:** Görüntü 224×224'e getirilir.

Ardından:

4. **Etiket kodlama:** Model metinle değil sayılarla çalışır. `LabelEncoder` sınıf adlarını sayıya çevirir (`airplanes -> 0`, `motorbikes -> 1`, `schooners -> 2`). Bu eşleme `le.pickle` dosyasına kaydedilir. Tahmin sırasında sayıyı tekrar sınıf adına çevirmek için gerekir.
5. **Eğitim/test ayırma:** Veri `train_test_split` ile %80 eğitim, %20 test olarak bölünür. `stratify` parametresi her sınıfın oranının iki kümede de aynı kalmasını sağlar.
6. Test görüntülerinin yolları `test_paths.txt` dosyasına yazılır. Inference adımında modelin hiç görmediği bu görüntüler kullanılır.

### Adım 6 – Dataset sınıfı ve model
**`CustomTensorDataset`:** PyTorch'un `DataLoader`'ı veriyi batch'ler hâlinde modele verir. Bunun için her örneğe (görüntü, etiket, kutu) üçlüsü olarak erişebilen bir `Dataset` sınıfı gerekir. Görüntüler bu sınıfta tensöre çevrilir ve ImageNet ortalama/standart sapmasıyla normalize edilir:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

*Neden?* ResNet50 ImageNet'te bu değerlerle normalize edilmiş görüntülerle eğitildi. Ona aynı biçimde hazırlanmış görüntü vermezsek öğrendiği özellikler doğru çalışmaz.

**`ObjectDetector`:** Model üç parçadan oluşur:

| Parça | Yapı | Görevi |
|---|---|---|
| Gövde | ResNet50 (dondurulmuş), son katmanı `nn.Identity()` | Görüntüden 2048 boyutlu özellik vektörü çıkarır |
| Kutu regresör kafası | `Linear(2048→128→64→32→4)` + `ReLU` + `Sigmoid` | 4 kutu koordinatını tahmin eder |
| Sınıflandırıcı kafa | `Linear(2048→512→512→3)` + `ReLU` + `Dropout` | Sınıf skorlarını üretir |

Bazı ayrıntılar:
- **`nn.Identity()`:** ResNet50'nin son katmanı ImageNet'in 1000 sınıfını tahmin eder. Bu katmanı, girdiyi değiştirmeden geçiren `Identity` ile değiştirerek gövdeden doğrudan özellik vektörünü alırız.
- **`Sigmoid`:** Çıktıyı 0–1 arasına sıkıştırır. Kutuları 0–1 arasına normalize ettiğimiz için tahminlerin de bu aralıkta olması gerekir.
- **`Dropout`:** Eğitim sırasında nöronların bir kısmını rastgele kapatır. Model belirli nöronlara aşırı bağımlı olamaz, bu da ezberlemeyi azaltır.
- **`requires_grad = False`:** Gövdenin ağırlıklarını dondurur. Böylece eğitimde yalnızca iki kafa güncellenir.

### Adım 7 – Eğitim
Her epoch'ta eğitim verisi batch'ler hâlinde dolaşılır. Her batch için:

```python
predBBoxes, predLogits = objectDetector(images)        # 1. İleri yayılım: tahmin üret
bboxLoss  = bboxLossFunc(predBBoxes, bboxes)            # 2. Kutu kaybı (MSE)
classLoss = classLossFunc(predLogits, labels)           #    Sınıf kaybı (Cross-Entropy)
totalLoss = (BBOX * bboxLoss) + (LABELS * classLoss)    #    Toplam kayıp

opt.zero_grad()       # 3. Önceki adımdan kalan gradyanları sıfırla
totalLoss.backward()  # 4. Geri yayılım: her ağırlığın kayba etkisini (gradyanı) hesapla
opt.step()            # 5. Ağırlıkları gradyanın tersi yönünde güncelle
```

Optimizasyon algoritması olarak **Adam** kullanılır. Adam, her ağırlık için öğrenme oranını otomatik ayarlayan, pratikte iyi çalışan bir yöntemdir.

Her epoch sonunda model test verisinde, ağırlıkları değiştirmeden değerlendirilir. Kayıp ve doğruluk değerleri grafik için saklanır.

Eğitim bitince model iki şekilde kaydedilir:
- `detector.pth`: Modelin tamamı (orijinal projedeki gibi).
- `detector_weights.pth`: Yalnızca ağırlıklar (`state_dict`). PyTorch'un önerdiği, sürüm değişikliklerine karşı daha dayanıklı yöntemdir.

### Adım 8 – Accuracy ve Loss grafikleri
Eğitim boyunca saklanan değerler epoch'a göre çizilir:
- Toplam kayıp (eğitim ve test)
- Kutu kaybı (eğitim ve test)
- Sınıflandırma doğruluğu (eğitim ve test)

Grafiklerin nasıl yorumlanacağı [Bölüm 8](#8-sonuçları-yorumlama)'de anlatılmıştır.

### Adım 9 – Sürüm hataları tablosu
Orijinal koda göre yapılan düzeltmeler listelenir. Ayrıntılar [Bölüm 5](#5-giderilen-sürüm-hataları)'tedir.

### Adım 10 – Inference (çıkarım)
Inference, eğitilmiş modeli yeni bir görüntü üzerinde kullanmaktır. `tahmin_et()` fonksiyonu bunu ekran veya pencere açmadan yapar ve sonucu değer olarak döndürür.

```python
etiket, olasilik = tahmin_et("/content/data/.../image_0001.jpg")
print(etiket, olasilik)
```

| | |
|---|---|
| Girdi | Görüntü dosyasının yolu |
| Çıktı | `(etiket, olasilik)`: tahmin edilen sınıf adı ve 0–1 arası olasılık |
| İsteğe bağlı | `kutu_dondur=True` verilirse piksel cinsinden kutu da döner: `(etiket, olasilik, (x1, y1, x2, y2))` |

Fonksiyonun içinde sırasıyla şunlar olur:

1. Model `detector.pth` dosyasından, etiket eşlemesi `le.pickle` dosyasından yüklenir.
2. Görüntü okunur ve orijinal boyutu (`w`, `h`) saklanır.
3. **Eğitimdeki ön işlemenin aynısı** uygulanır: BGR→RGB, 224×224, tensöre çevirme, normalizasyon. Ön işleme eğitimdekinden farklı olursa model yanlış tahmin yapar.
4. `unsqueeze(0)` ile tensöre batch boyutu eklenir. Model tek görüntü değil batch beklediği için `(3, 224, 224)` şekli `(1, 3, 224, 224)` olur.
5. Model `torch.inference_mode()` içinde çalıştırılır. Gradyan hesaplanmadığı için daha hızlıdır ve daha az bellek kullanır.
6. Skorlar Softmax ile olasılığa çevrilir, en yüksek olasılıklı sınıf seçilir.
7. Sınıf numarası `le.inverse_transform` ile tekrar sınıf adına çevrilir.
8. İstenirse 0–1 arasındaki kutu, görüntünün orijinal genişlik ve yüksekliğiyle çarpılarak piksele geri çevrilir.

Notebook'ta bu fonksiyon tek bir test görüntüsünde, ardından rastgele 10 test görüntüsünde denenir. Son olarak tüm test kümesindeki doğruluk hesaplanır. Tahmin edilen kutu, ekran açılmadan bir görüntü dosyasına da çizilebilir.

### Adım 11 – Dosyaları indirme
Model, ağırlıklar, `le.pickle`, grafik ve test listesi `output_model.zip` adıyla tek dosyada toplanıp bilgisayara indirilir.

---

## 5. Giderilen sürüm hataları

Orijinal kod 2021'de yazıldı. O zamandan beri kütüphanelerde yapılan değişiklikler nedeniyle kod güncel Colab'da doğrudan çalışmaz.

| No | Orijinal kod | Güncel ortamdaki sorun | Çözüm |
|---|---|---|---|
| 1 | `resnet50(pretrained=True)` | torchvision 0.13'ten itibaren `pretrained` parametresi kaldırıldı; önce uyarı, güncel sürümlerde hata verir. | `resnet50(weights=ResNet50_Weights.DEFAULT)` |
| 2 | `torch.load(MODEL_PATH)` | PyTorch 2.6 ile `weights_only=True` varsayılan oldu. Modelin tamamını yüklerken `UnpicklingError` alınır. | `torch.load(yol, weights_only=False)` kullanmak veya yalnızca `state_dict` kaydedip yüklemek |
| 3 | `cv2.imshow("Output", image)` | Colab'da ekran yoktur; `DisabledFunctionError` hatası verir. | Sonuç değer olarak döndürülür (`tahmin_et()`), gerekirse dosyaya çizilir. |
| 4 | `argparse` ile `--input` parametresi | Notebook hücresinde komut satırı argümanı verilemez; `SystemExit: 2` hatası alınır. | Argüman yerine normal bir Python fonksiyonu: `tahmin_et(goruntu_yolu)` |
| 5 | `num_workers=os.cpu_count()` | Colab'da "worker sayısı fazla" uyarısı ve takılma riski. | `num_workers=2` |
| 6 | Sabit `dataset/annotations/*.csv` yapısı | Kaggle'daki veri setinin klasör yapısı farklı. | CSV ve `.mat` formatlarını otomatik tanıyan okuyucu |
| 7 | `from imutils import paths` | Paket Colab'da kurulu gelmeyebilir; `ModuleNotFoundError`. | Python'un standart `glob` / `os.walk` modülleri |

**`weights_only=False` hakkında bir uyarı:** Bu seçenek dosyadaki her türlü Python nesnesinin çalıştırılmasına izin verir. Yalnızca kendi ürettiğiniz veya güvendiğiniz model dosyalarında kullanılmalıdır. Başkasından gelen dosyalar için `state_dict` yöntemi tercih edilmelidir.

---
