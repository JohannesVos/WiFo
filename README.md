# Autoencoder für die Anomalieerkennung

Dieses Repository enthält den Code und das Modelle für einen Convolutional Autoencoder (CAE), der für die Rekonstruktion von Bildern verwendet wird. Der Autoencoder wurde mit PyTorch implementiert und trainiert und anschließend für die Anomalieerkennung verwendet. Der Datensatz, der für dieses Projekt verwendet wurde findet sich unter dem folgendem Link [Screw (186 MB)](https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads). Ziel des Projektes ist es den Einfluss von Attribute Noise auf die Anomalieerkennung mithilfe eines CAE zu untersuchen. Hierzu wurden die Trainingsdatenmithilfe des Programms "Transformation.py" verunreinigt. 

## Inhaltsverzeichnis

- [Beschreibung](#beschreibung)
- [Voraussetzungen](#voraussetzungen)
- [Datenvorverarbeitung](#datenvorverarbeitung)
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

## Datenvorverarbeitung
Für die Datenvorverarbeitung und statistische Verunreinigung der Trainingsdaten wird die Datei [Transformation.py](https://github.com/JohannesVos/WiFo/blob/main/src/Transformation.py) benötigt. In diesem Code werden eine Reihe von Bildtransformationen auf eine Teilmenge von Trainingsbildern im angegebenen Eingabeordner angewendet. Die transformierten Bilder werden anschließend im angegebenen Ausgabeordner gespeichert und können für das Training des Autoencoders verwendet werden. Separat von dem Trainingsdatensatz, der nur intakte Schrauben enthält, stellt der Datensatz [Screw (186 MB)](https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads) auch gelabelte Testdaten zur Verfügung. Diese Testdaten können mithilfe der Datei [Test.py](https://github.com/JohannesVos/WiFo/blob/main/src/Test.py) verwendet werden, um die Leistungsfähigkeit des Autoencoders zu überprüfen.

## Verwendung

1. Stellen Sie sicher, dass Ihr Trainingsdatensatz und Testdatensatz ordnungsgemäß vorbereitet sind. Sie müssen die Pfade zu den Ordnern angeben, in denen die Bilder gespeichert sind.

2. Öffnen Sie die Datei [Train.py](https://github.com/JohannesVos/WiFo/blob/main/src/Train.py) und geben Sie den Pfad zu Ihrem Trainingsdatensatz und Testdatensatz im Abschnitt "Definieren des Trainingsdatensatzes und Testdatensatzes über die Pfade" an.

3. Sie können die Parameter für die Bildvorverarbeitung und das Training des Autoencoders im Code anpassen, wenn gewünscht. Dazu gehören IMG_RESIZE, BATCH_SIZE, NUM_EPOCHS, usw.

4. Führen Sie das Training aus, indem Sie die train.py Datei ausführen. Der Autoencoder wird auf dem Trainingsdatensatz trainiert, und der Fortschritt wird auf der Konsole ausgegeben. Das Modell wird auch regelmäßig auf dem Testdatensatz evaluiert und das beste Modell wird gespeichert. Wichtig ist zu beachten, dass alle Pfade korrekt angegeben sind.

5. Testen Sie das trainierte Modell mithilfe der Datei [Test.py](https://github.com/JohannesVos/WiFo/blob/main/src/Test.py) Beachten Sie hierbei das Sie die richtigen Pfade zu dem Modell und den Ordner der Testdaten angeben.

## Modell

Dieser Code definiert einen Autoencoder in PyTorch, der aus einem Encoder und einem Decoder besteht. Der Encoder komprimiert die Eingabedaten in eine kleinere Darstellung durch eine Reihe von Convolutional Neural Networks (CNNs), während der Decoder diese komprimierte Darstellung nimmt und versucht, die ursprünglichen Daten zu rekonstruieren.

Die Strukturähnlichkeitsmetrik (SSIM) wird zur Bewertung der Qualität der rekonstruierten Bilder verwendet. Im Gegensatz zum Mean Squared Error (MSE) berücksichtigt SSIM sowohl die Pixelunterschiede als auch die strukturellen Informationen in den Bildern, was zu einer besseren Übereinstimmung mit der menschlichen Wahrnehmung von Bildqualität führt. Daher kann die Verwendung von SSIM als Verlustfunktion dazu führen, dass der Autoencoder Bilder erzeugt, die natürlicher und realistischer aussehen. Die genaue Struktur ist im Codedatei [Train.py](https://github.com/JohannesVos/WiFo/blob/main/src/Train.py) wiederzufinden.

## Ergebnisse

Die Ergebnisse des Trainings und der Rekonstruktion werden im save_dir_Ordner gespeichert. Sie können die trainierten Modelle und die rekonstruierten Bilder in diesem Ordner finden.
Es werden die orginalen, rekonstruierten und SSIM Bilder nebeneinander angezeigt bzw. gespeichert.

![Alt-Text](https://github.com/JohannesVos/WiFo/blob/main/Beispiel.png)

## Autoren 

- Johannes Vos
- Khai-Phong Nguyen
- Felix Kirmaier
- Muhamed Saleh

## Lizenz


