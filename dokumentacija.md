# Prepoznavanje registarskih tablica - CNN vs CRNN

## 1. Opis projekta

Projekat se bavi automatskim prepoznavanjem teksta sa registarskih tablica koriscenjem dubokog ucenja. Implementirana su dva pristupa:

- **CNN** (Convolutional Neural Network) - klasifikacija pojedinacnih karaktera
- **CRNN** (Convolutional Recurrent Neural Network) - end-to-end prepoznavanje cele tablice

Koriscen dataset: European License Plates (Kaggle), 735 slika tablica.

## 2. CNN model

### Kako radi

CNN pristup se sastoji iz dva koraka:

1. **Segmentacija** - Slika tablice se obradjuje pomocu OpenCV (binarizacija, detekcija kontura) kako bi se izdvojili pojedinacni karakteri.
2. **Klasifikacija** - Svaki izdvojeni karakter (28x28 piksela, grayscale) se klasifikuje pomocu CNN mreze u jednu od 36 klasa (0-9, A-Z).

### Arhitektura

- 4 konvoluciona bloka (Conv2D + BatchNorm + ReLU + MaxPool + Dropout)
- Kanali: 32 -> 64 -> 128 -> 128
- Potpuno povezani slojevi na kraju za klasifikaciju
- Ukupno parametara: **573,316**

### Trening

- Funkcija gubitka: CrossEntropyLoss sa label smoothing (0.10) i tezinama klasa
- Optimizer: Adam (lr=1e-3, weight_decay=5e-4)
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)
- Early stopping sa patience=10

### Ogranicenja

CNN zavisi od kvaliteta segmentacije. Ako segmentacija ne uspe da pravilno izdvoji karakter, klasifikacija automatski gresi. Segmentacija je uspesna na 55.1% tablica (405 od 735).

## 3. CRNN model

### Kako radi

CRNN je end-to-end model koji prima celu sliku tablice i direktno predvidja sekvencu karaktera, bez potrebe za segmentacijom pojedinacnih karaktera.

1. **CNN enkoder** - Izvlaci vizuelne feature mape iz slike tablice (28x192 piksela)
2. **BiLSTM** - Dva sloja bidirekcionalnog LSTM-a obradjuju sekvencijalne feature-e duž horizontalne ose
3. **CTC dekoder** - Connectionist Temporal Classification omogucava obuku bez potrebe za poravnanjem (alignment) izmedju ulaza i izlaza

### Arhitektura

- CNN enkoder: 4 konvoluciona bloka (64 -> 128 -> 256 -> 256 kanala)
- CLAHE preprocesiranje za normalizaciju kontrasta
- Adaptive pooling za fiksni broj vremenskih koraka (T=48)
- BiLSTM: 2 sloja, hidden=256
- Spatial dropout (0.3) izmedju CNN i RNN dela
- Beam search dekodiranje (beam_width=10) za bolju predikciju sekvenci
- Ukupno parametara: **3,609,061**

### Trening

- Funkcija gubitka: CTC Loss
- Optimizer: Adam (lr=1e-3, weight_decay=5e-4)
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)
- Gradient clipping (max_norm=5.0) - neophodan za stabilnost RNN treninga
- Early stopping sa patience=10

### Prednosti

CRNN ne zahteva segmentaciju pojedinacnih karaktera. Direktno cita celu tablicu, sto ga cini robusnijim na razlicite formate tablica i probleme sa segmentacijom.

## 4. Kljucne razlike

| Aspekt | CNN | CRNN |
|---|---|---|
| Pristup | Segmentacija + klasifikacija | End-to-end (cela tablica) |
| Ulaz | Pojedinacni karakter (28x28) | Cela tablica (28x192) |
| Izlaz | Jedna klasa od 36 | Sekvenca karaktera |
| Funkcija gubitka | CrossEntropy | CTC Loss |
| Zavisnost od segmentacije | Da | Ne |
| Rukovanje ponovljenim karakterima | Nije problem (svaki char zasebno) | Beam search sa blank tokenom |
| Velicina modela | 573K parametara | 3.6M parametara |

## 5. Zajednicki parametri treninga

Radi fer poredjenja, oba modela koriste identicne hiperparametre:

| Parametar | Vrednost |
|---|---|
| Seed | 2003 |
| Batch size | 64 |
| Max epoha | 200 |
| Learning rate | 1e-3 |
| Weight decay | 5e-4 |
| Dropout rate | 0.5 |
| Early stop patience | 10 |
| Train/Test split | 80/20 |
| Train/Val split | 85/15 |
| Augmentacija | RandomAffine, RandomAutocontrast, RandomPerspective, RandomErasing |

## 6. Izlazi modela

Svaki model generise sledece fajlove:

**CNN:**
- `char_cnn_weights.pth` - nauceni tezinski parametri
- `loss_curves_cnn.png` - grafik train/val gubitka po epohama
- `cnn_metrics.csv` - sve metrike u CSV formatu

**CRNN:**
- `char_crnn_weights.pth` - nauceni tezinski parametri
- `loss_curves_crnn.png` - grafik train/val gubitka po epohama
- `crnn_metrics.csv` - sve metrike u CSV formatu

**Poredjenje:**
- `comparison_chart.png` - uporedni grafik metrika
- `comparison_per_class_f1.png` - F1 po klasama za oba modela

## 7. Rezultati i poredjenje metrika

### Metrike na nivou karaktera (test set)

| Metrika | CNN | CRNN | Bolji |
|---|---|---|---|
| Accuracy | 69.12% | 74.15% | CRNN |
| Precision (weighted) | 72.92% | 74.75% | CRNN |
| Recall (weighted) | 69.12% | 74.15% | CRNN |
| F1-Score (weighted) | 69.56% | 73.85% | CRNN |

### Metrike na nivou tablica (svih 735)

| Metrika | CNN | CRNN | Bolji |
|---|---|---|---|
| Plate exact match | 21.50% | 59.05% | CRNN |
| Mean CER | 0.3839 | 0.1190 | CRNN |
| Inference (ms) | 15.31 | 28.64 | CNN |

### Konfuzni parovi (F1 za cesto mesane karaktere)

| Par | CNN | CRNN | Bolji |
|---|---|---|---|
| B / 8 | 0.708 / 0.780 | 0.817 / 0.796 | CRNN |
| O / 0 | 0.133 / 0.186 | 0.286 / 0.825 | CRNN |
| S / 5 | 0.667 / 0.778 | 0.857 / 0.832 | CRNN |
| I / 1 | 0.000 / 0.825 | 0.364 / 0.727 | CRNN |

### Trening

| Detalj | CNN | CRNN |
|---|---|---|
| Early stop epoha | 71 | 200 (bez early stop) |
| Best val_loss | 2.2417 | 0.8849 |

## 8. Zakljucak

CRNN model pokazuje znacajno bolje rezultate u poredjenju sa CNN modelom na svim metrikama kvaliteta:

- **+5% tacnost na nivou karaktera** (74.15% vs 69.12%)
- **+37.5 procentnih poena tacnost na nivou tablica** (59.05% vs 21.50%)
- **3.2x manji CER** (0.119 vs 0.384)

Glavna prednost CRNN-a je end-to-end pristup koji eliminise zavisnost od segmentacije. CNN prednjaci jedino u brzini inferencije (15ms vs 29ms) i velicini modela (573K vs 3.6M parametara).

Za prakticnu primenu prepoznavanja registarskih tablica, CRNN je bolji izbor jer znacajno preciznije prepoznaje cele tablice, sto je krajnji cilj sistema.
