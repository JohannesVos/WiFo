# Autoencoder für die Anomalieerkennung

Dieses Repository enthält den Code und das Modell für einen Convolutional Autoencoder (CAE), der für die Rekonstruktion von Bildern verwendet wird. Der Autoencoder wurde mit PyTorch implementiert und trainiert und anschließend für die Anomalieerkennung verwendet. Der Datensatz, der für dieses Projekt verwendet wurde findet sich unter dem folgendem Link [Screw (186 MB)](https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads).

## Inhaltsverzeichnis

- [Beschreibung](#beschreibung)
- [Voraussetzungen](#voraussetzungen)
- [Verwendung](#verwendung)
- [Modell](#modell)
- [Ergebnisse](#ergebnisse)
- [Autoren](#autoren)
- [Lizenz](#lizenz)

## Beschreibung

Ein Autoencoder ist ein neuronales Netzwerk, das verwendet wird, um Daten zu komprimieren und zu rekonstruieren. Im Falle eines CAE wird der Autoencoder speziell für die Verarbeitung von Bildern entwickelt. Der Encoder nimmt ein Eingangsbild entgegen und komprimiert es auf eine niedrigdimensionale Darstellung, die als Latent Space bezeichnet wird. Der Decoder nimmt die Latent Space-Repräsentation entgegen und rekonstruiert das Bild aus dieser Darstellung.

Dieses Projekt verwendet einen CAE, um Bilder zu rekonstruieren. Der Autoencoder wird mit einem Trainingsdatensatz trainiert und anschließend verwendet, um Testbilder zu rekonstruieren. Das Modell wird mit dem SSIM-Verlust (Structural Similarity Index) trainiert, der die strukturelle Ähnlichkeit zwischen den rekonstruierten Bildern und den Originalbildern misst.

## Voraussetzungen

Um den Code in diesem Repository auszuführen, sind die folgenden Voraussetzungen erforderlich:

- Python 3.7 oder höher
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- scikit-image
- pytorch-msssim

## Verwendung

Um das Modell zu trainieren, führen Sie den folgenden Befehl aus:
```python Train.py```

Um das Modell zu testen, führen Sie den folgenden Befehl aus:
```python Train.py``` 

## Modell

Dieser Code definiert einen Autoencoder in PyTorch, der aus einem Encoder und einem Decoder besteht. Der Encoder komprimiert die Eingabedaten in eine kleinere Darstellung durch eine Reihe von Convolutional Neural Networks (CNNs), während der Decoder diese komprimierte Darstellung nimmt und versucht, die ursprünglichen Daten zu rekonstruieren.

Die Strukturähnlichkeitsmetrik (SSIM) wird zur Bewertung der Qualität der rekonstruierten Bilder verwendet. Im Gegensatz zum Mean Squared Error (MSE) berücksichtigt SSIM sowohl die Pixelunterschiede als auch die strukturellen Informationen in den Bildern, was zu einer besseren Übereinstimmung mit der menschlichen Wahrnehmung von Bildqualität führt. Daher kann die Verwendung von SSIM als Verlustfunktion dazu führen, dass der Autoencoder Bilder erzeugt, die natürlicher und realistischer aussehen. Die genaue Struktur ist im Codedatei Train.py wiederzufinden.

## Ergebnisse

Die Ergebnisse des Trainings und der Rekonstruktion werden im save_dir-Ordner gespeichert. Sie können die trainierten Modelle und die rekonstruierten Bilder in diesem Ordner finden.
Es werden die orginalen(jupyterlab), rekonstruierten und SSIM Bilder nebeneinander angezeigt bzw. gespeichert.
![Alt-Text]([https://github.com/JohannesVos/WiFo/blob/main/Screenshot%202023-07-24%20195410.png]

## Autoren 

- Johannes Vos
- Khai-Phong Nguyen
- Felix Kirmaier
- Muhammed

## Lizenz

