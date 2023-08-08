import os
import random
import shutil
from PIL import Image
from torchvision import transforms

# Schärfefaktor
kernel_size = 15

# Definiere deine Transformationen
transformations = [
    #transforms.GaussianBlur(kernel_size, sigma=(3.0, 3.0)),   # Gausscher Weichzeichner
    #transforms.ColorJitter(brightness=0.4),                   # Helligkeitsvariation
    #transforms.deleteRandomPixels(66)                         # Zufälliges Entfernen von Pixeln
    transforms.Grayscale(),                                    # Umwandlung in Graustufen
    transforms.Resize(256)                                     # Größenänderung auf 256x256 Pixel
]

# Gib den Pfad zum Eingabeordner an
input_folder = r"C:\Users\johan\Desktop\bottle\test_merged"

# Gib den Pfad zum Ausgabeordner für die transformierten Bilder an
output_folder = r"C:\Users\johan\Desktop\bottle\test_merged_transformed"

# Erstelle den Ausgabeordner, wenn er noch nicht existiert
os.makedirs(output_folder, exist_ok=True)

# Erhalte eine Liste der Bilddateien im Eingabeordner
image_files = [file for file in os.listdir(input_folder) if file.endswith(('.jpg', '.jpeg', '.png'))]

# Berechne die Anzahl der zu transformierenden Bilder (x% des Datensatzes)
num_images = int(len(image_files) * 1)

# Wähle zufällig Bilder für den Teil-Datensatz aus
subset_image_files = random.sample(image_files, num_images)

# Wende die Transformationen auf den Teil-Datensatz an und speichere sie im Ausgabeordner
for image_file in subset_image_files:
    # Lade das Bild
    image_path = os.path.join(input_folder, image_file)
    image = Image.open(image_path)
    transformed_image = image

    # Wende die Transformationen nacheinander auf das Bild an
    for transformation in transformations:
        transformed_image = transformation(transformed_image)

    # Speichere das transformierte Bild im Ausgabeordner
    output_image_path = os.path.join(output_folder, image_file)
    transformed_image.save(output_image_path)

# Kopiere die restlichen Bilder vom Eingabeordner in den Ausgabeordner
for image_file in image_files:
    if image_file not in subset_image_files:
        input_image_path = os.path.join(input_folder, image_file)
        output_image_path = os.path.join(output_folder, image_file)
        shutil.copyfile(input_image_path, output_image_path)
