# Pokemon - GAI
**Group 1:** Lisa Rost, Merna Zaher, Lena Huber, Marcel Dielacher

## Use Case:

Für unser Projekt haben wir drei spezifische Anwendungsfälle definiert:

**Generierung von Daten mit CTGAN:** Wir nutzen Conditional Generative Adversarial Networks (CTGAN), um synthetische Daten zu erstellen. Dies hilft uns, Muster und Szenarien zu simulieren, die in den ursprünglichen Daten nicht ausreichend vertreten sind.

**Mit Tensforflow ein eigenes GAN zu programmieren:** Dieser Output wird dann mit dem CTGAN Daten verglichen un evaluiert.

**Generierung von Namen mit einem LLM:** Auf Basis der durch CTGAN generierten Daten verwenden wir ein Large Language Model (LLM), um Namen für die synthetischen Pokémon-Daten zu erstellen. Dieser Prozess ermöglicht es, kreative und realistisch klingende Namen automatisiert zu generieren.

## Data Description:
**Source:**: https://www.kaggle.com/datasets/abcsds/pokemon
**Llama 2 7b:** https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF

**Datenqualität:** 
Die Datenqualität des Rohdatensatzes war generell gut, jedoch waren die Namen der Einträge sehr inkonsistent. Weitere Details dazu finden Sie im Kapitel zur Datenvorbereitung.

## Process:

### Data Preperation:
Im Rahmen der Datenvorbereitung wurde zunächst der Rohdatensatz eingelesen. Um die Daten übersichtlicher und effizienter für die weitere Analyse zu gestalten, haben wir uns entschieden, bestimmte Spalten zu entfernen. Insbesondere die Spalten Total und #, welche für unsere Analysezwecke nicht relevant waren, wurden aus dem Datensatz entfernt.

Ein weiterer wichtiger Schritt in der Datenvorbereitung war die Bearbeitung der Pokémon-Namen. Viele Namen im ursprünglichen Kaggle-Datensatz waren inkonsistent formatiert, enthielten unnötige Wiederholungen oder Abkürzungen, die nicht standardisiert waren. Um dieses Problem zu adressieren, wurden die Namen mithilfe von Funktionen automatisiert bereinigt. Zusätzlich wurde eine manuelle Nachbearbeitung durchgeführt, um spezielle Fälle zu korrigieren, die nicht vollständig durch die automatisierten Prozesse abgedeckt werden konnten.

Diese Schritte stellen sicher, dass der Datensatz sauber, konsistent und bereit für die nachfolgende Analyse ist. Der vorbereitete Datensatz wird in der Datei **preprocessed_pokemon.csv** gespeichert, die für alle weiteren Analyseschritte verwendet wird. 
Zu finden ist dieses Kapitel unter: **"00_data_prep_cleaning_understanding.ipynb"**

Ein zusätzlicher Aspekt unseres Projekts war die Zusammenführung der Evolutionsstufen der Pokémon, was sich als besonders herausfordernd erwies. Der Versuch, die Daten mit ChatGPT 4 zu erfassen, führte nicht zum gewünschten Erfolg. Wir experimentierten auch mit dem Notebook **[EINFÜGEEN !!!!!!!!!]** und setzten CoPilot ein, um die Daten zu rekonstruieren, was überraschend gut gelang. Allerdings gab es Formatierungsprobleme bei einigen Pokémon. Anschließend haben wir die Daten mit einer weiteren Datei **einfügen !!!!!!!!** zusammengeführt um die richtigen Evolutionsstufen zu mergen.

### Data Understanding:
In diesem Kapitel widmen wir uns der eingehenden Analyse der Rohdaten. Ziel ist es, ein tiefes Verständnis für die vorhandenen Datenstrukturen und die in den Daten enthaltenen Informationen zu entwickeln. Dies umfasst die Untersuchung von Datenverteilungen, das Erkennen von Mustern und das Identifizieren von Zusammenhängen zwischen verschiedenen Datenpunkten. Zu finden ist dieses Kapitel unter: **"00_data_prep_cleaning_understanding.ipynb"**

**Hauptaktivitäten:**

**1. Deskriptive Statistik:** Wir berechnen grundlegende statistische Kennzahlen wie Mittelwert, Median, Modus, Standardabweichung usw., um ein Gefühl für die Verteilung der Daten zu bekommen.
Visualisierung: Durch den Einsatz verschiedener Diagramme und Plots (z.B. Histogramme, Boxplots, Scatterplots) visualisieren wir die Daten, um Trends, Ausreißer und Gruppierungen besser erkennen zu können.

**2. Korrelationsanalyse:** Wir untersuchen die Beziehungen zwischen verschiedenen Variablen, um zu verstehen, welche Merkmale miteinander korrelieren. Dies hilft uns, Hypothesen für weitergehende Analysen zu formulieren.

**3. Erkennung von Anomalien:** Eventuelle Datenanomalien oder ungewöhnliche Muster werden identifiziert, die spezielle Aufmerksamkeit erfordern oder auf Datenbereinigungsbedarf hinweisen könnten.
Ergebnisse dieses Kapitels:

Zu finden ist dieses Kapitel unter: **"00_data_prep_cleaning_understanding.ipynb"**

### Model:
#### CTGAN:

#### Tensorflow GAN:

### Namensgenerierung der Pokemons:
#### GPT-3.5-turbo:
Dieses Modell wurde aufgrund seiner neuesten Architektur und optimierten Performance für die Generierung kreativer und konsistenter Pokemon-Namen ausgewählt. Die Basis der Namensgenerierung bilden die statistischen Daten der Pokemon, wie Typ, Angriffswerte, Verteidigungswerte usw.

#### Llama 2 7b:
Das LLaMA 2 7B Modell wurde ebenfalls eingesetzt, um die Ergebnisse zu vergleichen. Obwohl es hochentwickelt ist, zeigten unsere Tests, dass das Prompting mit LLaMA 2 7B nicht immer effektiv funktionierte. Teilweise wurden passende Namen generiert, während in anderen Fällen keine sinnvollen Ergebnisse produziert wurden.

#### Ergebnisse und Diskussion:
Die durchgeführten Experimente zeigten, dass GPT-3.5-turbo konsistentere und kreativere Namen lieferte im Vergleich zu LLaMA 2 7B. Die Herausforderungen bei der Verwendung von LLaMA 2 7B könnten auf verschiedene Faktoren zurückzuführen sein, einschließlich der Art des Prompting und der Modellkonfiguration. Diese Ergebnisse betonen die Wichtigkeit der Modellauswahl und -anpassung für spezifische Aufgaben im Bereich des Natural Language Processing.

Zu finden ist dieses Kapitel unter: **"pokemon_namen_generierung.ipynb"**

### Evaluation:
