import os
import shutil
from math import floor
from math import log2
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import torchvision.transforms as transforms
from PIL import Image
        
# Definiere die Bildgröße und andere Parameter
pixel = 256
kernel_size = 4
padding = 1
stride = 2
output_padding = 0

# Definiere die Anzahl der Cluster
cluster = 2

# Definiere den Pfad zum Speichern der Bilder
save_images_dir = '/home/io556773/jupyterlab/Anomaly_detection_MVtec/Clusters'

# Definiere den Pfad zu den Testdaten
test_data_path = '/home/io556773/jupyterlab/JMKF_Autoencoder/Data/MVtec_orginal_test'
image_files = [os.path.join(test_data_path, file) for file in os.listdir(test_data_path) if file.endswith('.png')]

# Definiere das Autoencoder-Modell
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Definiere den Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 500, 3, stride=1, padding=0)
        )

        # Definiere den Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(500, 128, 3, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Lade das Autoencoder-Modell
model = Autoencoder()

checkpoint = torch.load('/home/io556773/jupyterlab/Anomaly_detection_MVtec/Model/sharpness/100/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(model)

# Definiere die Transformation für die Testdaten
data_transforms_standard = transforms.Compose([
    transforms.Resize(pixel),
    transforms.ToTensor(),
    transforms.Grayscale(),
])

def reconstructed_images(model, image_files, save_images_dir):
    if os.path.exists(save_images_dir):
        shutil.rmtree(save_images_dir)

    if not os.path.exists(save_images_dir):
        os.makedirs(save_images_dir)

    mse_list = []

    for image_path in image_files:
        # Lade das Bild
        img = cv2.imread(image_path)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (256, 256))

        # Konvertiere das Bild in das passende Format für den Autoencoder
        img = Image.fromarray(image)
        img = data_transforms_standard(img)
        img = img.unsqueeze(0)

        # Rekonstruiere das Bild
        output = model(img)
        output = output.squeeze(0).squeeze(0)
        output = output.detach().numpy()

        # Speichere das rekonstruierte Bild
        save_path = os.path.join(save_images_dir, os.path.basename(image_path))
        cv2.imwrite(save_path, (output * 255).astype(np.uint8))

        # Berechne den MSE zwischen dem rekonstruierten Bild und dem Originalbild
        mse = np.square(np.subtract(image, output * 255)).mean()
        mse_list.append(mse)

    return mse_list

mse_list = reconstructed_images(model, image_files, save_images_dir)

def Pixel_difference(model, image_files, save_images_dir):
    max_diff_list = []
    
    for image_path in image_files:
        # Lade das Bild
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (256, 256))

        # Konvertiere das Bild in das passende Format für den Autoencoder
        img = Image.fromarray(img)
        img = data_transforms_standard(img)
        img = img.unsqueeze(0)

        # Rekonstruiere das Bild
        output = model(img)
        output = output.squeeze(0).squeeze(0)
        output = output.detach().numpy()

        # Berechne den maximalen Pixelunterschied
        max_diff = 0
        for i in range(img.size()[2]):
            for j in range(img.size()[3]):
                pixel_diff = abs(img[0, 0, i, j] - output[i, j]) * 255
                if pixel_diff > max_diff:
                    max_diff = pixel_diff

        print("Maximaler Pixelunterschied:", max_diff)
        max_diff_list.append(max_diff)
        
    min_max_diff = np.min(max_diff_list)
    
    print("Minimaler maximaler Pixelunterschied:", min_max_diff)
    
    return min_max_diff

min_max_diff = Pixel_difference(model, image_files, save_images_dir)

def Clustering(min_max_diff, save_img_dir):
    errors = []
    
    print('_____________________________________', min_max_diff)
    
    for image_path in image_files:
        # Lade das Bild
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (256, 256))

        # Konvertiere das Bild in das passende Format für den Autoencoder
        img = Image.fromarray(img)
        img = data_transforms_standard(img)
        img = img.unsqueeze(0)

        # Rekonstruiere das Bild
        output = model(img)
        output = output.squeeze(0).squeeze(0)
        output = output.detach().numpy()

        # Berechne den absoluten Fehler für jedes Pixel und speichere ihn in einer Liste
        loss_list = []
        for i in range(img.size()[2]):
            for a in range(img.size()[3]):
                loss = abs(img[0, 0, i, a] - output[i, a]) * 255
                loss_list.append(loss.item())

        # Summiere alle Fehler auf, die größer als der minimale maximale Pixelunterschied sind
        error = 0
        for loss in loss_list:
            if loss > min_max_diff:
                error += loss  

        errors.append(error)
    
    max_error_threshold = np.mean(errors) * 0.4
    
    broke_folder = os.path.join(save_img_dir, 'broke')
    if os.path.exists(broke_folder):
        shutil.rmtree(broke_folder)

    # Lösche 'good'-Ordner, falls er existiert
    good_folder = os.path.join(save_img_dir, 'good')
    if os.path.exists(good_folder):
        shutil.rmtree(good_folder)

    # Erstelle 'broke'- und 'good'-Ordner
    os.makedirs(broke_folder)
    os.makedirs(good_folder)
    
    for image_path, error in zip(image_files, errors):
        # Speichere das Bild im entsprechenden Ordner basierend auf dem Fehler
        if error > max_error_threshold:
            # Erstelle 'broke'-Ordner, falls er nicht existiert
            broke_folder = os.path.join(save_img_dir, 'broke')
            os.makedirs(broke_folder, exist_ok=True)
            shutil.copy(image_path, broke_folder)  # Kopiere das Bild in den 'broke'-Ordner
        else:
            # Erstelle 'good'-Ordner, falls er nicht existiert
            good_folder = os.path.join(save_img_dir, 'good')
            os.makedirs(good_folder, exist_ok=True)
            shutil.copy(image_path, good_folder)  # Kopiere das Bild in den 'good'-Ordner

    return errors

errors = Clustering(min_max_diff, save_images_dir)

def count_specific_image_names_in_folders(save_images_dir):
    categories = ['good', 'manipulated_front', 'scratch_head', 'thread_side', 'thread_top', 'scratch_neck']
    count_data = []
    
    folder = os.path.join(save_images_dir, 'broke')
    
    # Zähle die Anzahl der Bilder in jeder Kategorie im 'broke'-Ordner
    count_row = []
    for category in categories:
        count = 0
        for filename in os.listdir(folder):
            if category in filename:
                count += 1
        count_row.append(count)
    count_data.append(count_row)
    
    folder = os.path.join(save_images_dir, 'good')
    # Zähle die Anzahl der Bilder in jeder Kategorie im 'good'-Ordner
    count_row = []
    for category in categories:
        count = 0
        for filename in os.listdir(folder):
            if category in filename:
                count += 1
        count_row.append(count)
    count_data.append(count_row)

    # Erstelle ein pandas DataFrame mit den Zähldaten
    count_df = pd.DataFrame(count_data, columns=categories)
    
    # Speichere das DataFrame als CSV-Datei
    count_df.to_csv('count_df.csv', index=False)
    
    # Gib das DataFrame aus
    print(count_df)

    return count_df

count_specific_image_names_in_folders(save_images_dir)

def sort_images(errors, image_files, save_images_dir):
    cluster_ranges = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]

    for image_path in image_files:
        # Lade das Bild
        img = cv2.imread(image_path)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (256, 256))

        # Konvertiere das Bild in das passende Format für den Autoencoder
        img = Image.fromarray(image)
        img = data_transforms_standard(img)
        img = img.unsqueeze(0)

        # Rekonstruiere das Bild
        output = model(img)
        output = output.squeeze(0).squeeze(0)
        output = output.detach().numpy()

        # Cluster die Bilder in zwei Ordner mit Fehlern
        
        mse = np.mean(np.square(np.subtract(image, output * 255)))
        
        cluster_index = None
        for i, (cluster_min, cluster_max) in enumerate(cluster_ranges):
            if cluster_min <= mse <= cluster_max:
                cluster_index = i
                break

        if cluster_index is not None:
            # Speichere das Bild im entsprechenden Cluster-Ordner
            cluster_dir = os.path.join(save_images_dir, 'cluster_' + str(cluster_index))
            if not os.path.exists(cluster_dir):
                os.makedirs(cluster_dir)
            cv2.imwrite(os.path.join(cluster_dir, os.path.basename(image_path)), image)

# sort_images(errors, image_files, save_images_dir)
