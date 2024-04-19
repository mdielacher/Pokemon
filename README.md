# Pokemon - GAI
**Group 1:** Lisa Rost, Merna Zaher, Lena Huber, Marcel Dielacher

## Use Case:

Für unser Projekt haben wir drei spezifische Anwendungsfälle definiert:

**Generierung von Daten mit CTGAN:** Wir nutzen Conditional Generative Adversarial Networks (CTGAN), um synthetische Daten zu erstellen. Dies hilft uns, Muster und Szenarien zu simulieren, die in den ursprünglichen Daten nicht ausreichend vertreten sind.

**Generierung von Namen mit einem LLM:** Auf Basis der durch CTGAN generierten Daten verwenden wir ein Large Language Model (LLM), um Namen für die synthetischen Pokémon-Daten zu erstellen. Dieser Prozess ermöglicht es, kreative und realistisch klingende Namen automatisiert zu generieren.

**Überprüfung durch Generator und Diskriminator:** Um die Qualität der generierten Daten zu testen, setzen wir einen Generator und Diskriminator ein. Diese prüfen, ob das Generative Adversarial Network (GAN) in der Lage ist, zwischen echten und synthetischen Pokémon-Daten zu unterscheiden.

## Data Description:
**Source:**: https://www.kaggle.com/datasets/abcsds/pokemon

**Datenqualität:** 
Die Datenqualität des Rohdatensatzes war generell gut, jedoch waren die Namen der Einträge sehr inkonsistent. Weitere Details dazu finden Sie im Kapitel zur Datenvorbereitung.


## Process:

### Data Preperation:
Im Rahmen der Datenvorbereitung wurde zunächst der Rohdatensatz eingelesen. Um die Daten übersichtlicher und effizienter für die weitere Analyse zu gestalten, haben wir uns entschieden, bestimmte Spalten zu entfernen. Insbesondere die Spalten Total und #, welche für unsere Analysezwecke nicht relevant waren, wurden aus dem Datensatz entfernt.

Ein weiterer wichtiger Schritt in der Datenvorbereitung war die Bearbeitung der Pokémon-Namen. Viele Namen im ursprünglichen Kaggle-Datensatz waren inkonsistent formatiert, enthielten unnötige Wiederholungen oder Abkürzungen, die nicht standardisiert waren. Um dieses Problem zu adressieren, wurden die Namen mithilfe von Funktionen automatisiert bereinigt. Zusätzlich wurde eine manuelle Nachbearbeitung durchgeführt, um spezielle Fälle zu korrigieren, die nicht vollständig durch die automatisierten Prozesse abgedeckt werden konnten.

Diese Schritte stellen sicher, dass der Datensatz sauber, konsistent und bereit für die nachfolgende Analyse ist. Der vorbereitete Datensatz wird in der Datei preprocessed_pokemon.csv gespeichert, die für alle weiteren Analyseschritte verwendet wird.

Ein weiterer Schritt war es die Evolutionsstufen der Pokemons zu mergen, dies stellte eine Riesen Herausforderung dar.

Zu finden ist dieses Kapitel unter: **"00_data_prep_cleaning_understanding.ipynb"**

### Data Understanding:
In diesem Kapitel widmen wir uns der eingehenden Analyse der Rohdaten. Ziel ist es, ein tiefes Verständnis für die vorhandenen Datenstrukturen und die in den Daten enthaltenen Informationen zu entwickeln. Dies umfasst die Untersuchung von Datenverteilungen, das Erkennen von Mustern und das Identifizieren von Zusammenhängen zwischen verschiedenen Datenpunkten.

**Hauptaktivitäten:**

**1. Deskriptive Statistik:** Wir berechnen grundlegende statistische Kennzahlen wie Mittelwert, Median, Modus, Standardabweichung usw., um ein Gefühl für die Verteilung der Daten zu bekommen.
Visualisierung: Durch den Einsatz verschiedener Diagramme und Plots (z.B. Histogramme, Boxplots, Scatterplots) visualisieren wir die Daten, um Trends, Ausreißer und Gruppierungen besser erkennen zu können.

**2. Korrelationsanalyse:** Wir untersuchen die Beziehungen zwischen verschiedenen Variablen, um zu verstehen, welche Merkmale miteinander korrelieren. Dies hilft uns, Hypothesen für weitergehende Analysen zu formulieren.

**3. Erkennung von Anomalien:** Eventuelle Datenanomalien oder ungewöhnliche Muster werden identifiziert, die spezielle Aufmerksamkeit erfordern oder auf Datenbereinigungsbedarf hinweisen könnten.
Ergebnisse dieses Kapitels:

Zu finden ist dieses Kapitel unter: **"00_data_prep_cleaning_understanding.ipynb"**

### Model:

## Namensgenerierung der Pokemons:
### Evaluation: