# WellSense Watch – inteligentne zdrowie na Twoim nadgarstku

**WellSense Watch** to nowoczesna aplikacja zdrowotna, która wykorzystuje dane ze smartwatcha oraz zaawansowane modele sztucznej inteligencji (AI), aby w czasie rzeczywistym wykrywać i monitorować potencjalne oznaki chorób – zanim pojawią się poważne objawy.

## Sztuczna inteligencja w służbie zdrowia

Obecnie wykorzystujemy model AI wytrenowany na publicznie dostępnych danych testowych:  
https://data.mendeley.com/datasets/dzz48mvjht/1/files/2ee83017-9af4-4618-b340-b8594accc63f

Model osiąga 91% skuteczności w przewidywaniu, czy osoba – na podstawie danych ze smartwatcha – może wykazywać symptomy chorobowe.

## Jak to działa?

Aplikacja zbiera dane z wbudowanych sensorów smartwatcha, takich jak:
- tętno,
- poziom tlenu we krwi,
- sen,
- aktywność fizyczna,
- poziom stresu.

Dane te są przesyłane do serwera, gdzie model AI dokonuje analizy i zwraca wynik predykcji. Każda przesyłana paczka danych zawiera również nazwę użytego modelu.

## Plany na przyszłość

W przyszłości planujemy rozwój nowych, bardziej precyzyjnych modeli AI.  
Aby to osiągnąć, konieczne będzie:
- zebranie większej liczby danych od użytkowników,
- weryfikacja danych przez specjalistów medycznych.

## Instrukcja użycia

### 1. Sklonuj repozytorium

```bash
git clone https://github.com/PiotrekPrezydent/Hackcarpathia.git
cd Hackcarpathia
```

### 2. Utwórz i aktywuj środowisko wirtualne

#### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows (cmd)

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Zainstaluj zależności

```bash
pip install -r requirements.txt
```

### 4. Uruchomienie modelu AI i serwera

#### Trening modelu

```bash
python heart_model.py
```

#### Uruchomienie serwera

```bash
python server.py
```

### 5. Testowanie działania

Aby wysłać przykładowe dane smartwatcha do serwera, uruchom:

```bash
python sample_smartwach_data.py
```

**Uwaga:** przed uruchomieniem tego skryptu upewnij się, że serwer działa.
