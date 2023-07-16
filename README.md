# Autoencoder für die Anomalieerkennung

Dieses Repository enthält den Code und das Modell für einen Convolutional Autoencoder (CAE), der für die Rekonstruktion von Bildern verwendet wird. Der Autoencoder wurde mit PyTorch implementiert und trainiert und anschließend für die Anomalieerkennung verwendet.

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
python Train.py

## Modell

Das Modell besteht aus einem Encoder und einem Decoder. Der Encoder besteht aus mehreren Convolutional-Schichten, die das Bild schrittweise komprimieren. Der Decoder besteht aus Transposed Convolutional-Schichten, die das Bild schrittweise rekonstruieren. Die genauen Architekturdetails finden Sie im Code.

## Ergebnisse

Die Ergebnisse des Trainings und der Rekonstruktion werden im save_dir-Ordner gespeichert. Sie können die trainierten Modelle und die rekonstruierten Bilder in diesem Ordner finden.

## Autoren 

- Johannes Vos
- Khai-Phong Nguyen
- Felix Kirmaier
- Muhammed

## Lizenz

