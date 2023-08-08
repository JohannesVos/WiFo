# Import der erforderlichen Bibliotheken
import os
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Compose, Grayscale, Resize

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path

from skimage.metrics import structural_similarity as ski_ssim
from pytorch_msssim import SSIM, MS_SSIM

# Die Variable "device" wird entweder auf "cuda" (GPU) oder "cpu" (CPU) gesetzt, je nach Verfügbarkeit einer CUDA-fähigen GPU.
# Dies stellt sicher, dass der Code auf der verfügbaren Hardware ausgeführt wird und die Berechnungen auf die GPU verschoben werden können, falls verfügbar.
device = "cuda" if torch.cuda.is_available() else "cpu"

print(torch.__version__)
print(torch.backends.cudnn.enabled)

# Definition der Parameter
NUM_WORKERS = os.cpu_count()
IMG_RESIZE = 256  # Verwendete Bildgröße ist 256*256 Pixel
BATCH_SIZE = 10  # Verwendete Batchgröße ist 10
NUM_EPOCHS = 400  # Trainingslänge ist 400 Epochen

# Definieren der Bildvorverarbeitung
data_transforms_standard = Compose([
    transforms.Resize(size=(IMG_RESIZE, IMG_RESIZE)),  # Verkleinern des Originalbildes von 1024*1024 Pixeln auf 256*256 Pixel
    transforms.Grayscale(1),  # Verkleinern des Originalbildes von RGB zu 1-dimensionalen Bildern in Graustufen
    transforms.RandomHorizontalFlip(p=0.5),  # Zufälliges Drehen des Bildes (Data Augmentation)
    transforms.RandomVerticalFlip(p=0.5),  # Zufälliges Drehen des Bildes (Data Augmentation)
    transforms.ToTensor()
])


# Definition des Datensatzes
class CustomDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_filenames = [filename for filename in os.listdir(folder_path) if
                                os.path.isfile(os.path.join(folder_path, filename))]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_filenames[idx])
        image = Image.open(img_path).convert('L')  # In Graustufen umwandeln
        if self.transform:
            image = self.transform(image)
        return image


# Definieren des Trainingsdatensatzes und Testdatensatzes über die Pfade
train_data_folder = '/Path_to_Training_Data'
test_data_folder = '/Path_to_Test_Data'

# Laden der Trainingsbilder aus dem Ordner und Erstellen des Datasets
train_dataset = CustomDataset(train_data_folder, transform=data_transforms_standard)

# Laden der Testbilder aus dem Ordner und Erstellen des Datasets
test_dataset = CustomDataset(test_data_folder, transform=data_transforms_standard)

# Erstellen der Trainings- und Testdatenlader, um die Daten in Batches aufzuteilen und die Effizienz des Trainings zu steigern
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH_SIZE,
                             num_workers=NUM_WORKERS,
                             shuffle=False)

# Überprüfen der Eingabebilder, um die geeigneten Aktivierungsfunktionen im Modell verwenden zu können
images = next(iter(train_dataloader))
print(f"Bildform: {images.shape}\n Minimum: {torch.min(images)}\n Maximum: {torch.max(images)}")


# Definition des CAE-Autoencoders
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Definition des Encoders
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
        # Definition des Decoders
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


# Anzeigen der Struktur vor dem Training
model = Autoencoder()
model = model.to(device)
print(model)

# Der SSIM-Verlust misst die strukturelle Ähnlichkeit zwischen den rekonstruierten Bildern und den Originalbildern.
# Die SSIM-Klasse wird aus dem Modul "pytorch_msssim" importiert und ermöglicht die Berechnung des SSIM-Verlusts.
criterion = SSIM(win_sigma=1.5, data_range=1, size_average=True, channel=1)

# Initialisierung des Adam-Optimierers
optimizer = torch.optim.Adam(params=model.parameters(), lr=2e-4)

loss_fn = criterion


# Definition der Trainingsschritte
def train_step(model, dataloader, loss_fn, optimizer, epoch):
    model.train()
    train_loss = 0
    best_loss = float('inf')  # Initialisiere die beste Verlustvariable

    for batch, img in enumerate(dataloader):
        img = img.to(device)
        recon = model(img)
        loss = 1 - loss_fn(recon, img)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_train_loss = train_loss / len(dataloader)
    train_ssim = (1 - avg_train_loss) * 100  # Berechne SSIM korrekt

    return train_loss, train_ssim


# Definition der Testschritte
def test_step(model, dataloader, loss_fn, epoch):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for batch, img in enumerate(dataloader):
            img = img.to(device)
            recon = model(img)
            loss = 1 - loss_fn(recon, img)
            test_loss += loss.item()

        test_loss = test_loss / len(dataloader)
        test_ssim = (1 - test_loss) * 100  # Berechne SSIM korrekt

    return test_loss, test_ssim


# Training des Modells, durch das Speichern mit Hilfe von Checkpoints ist es möglich, das Training abzubrechen und wieder aufzunehmen
def train(model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs):
    # Definition des Pfads, in den das Modell gespeichert werden soll
    save_dir = '/Path_to_safe_dir' # Pfad anpassen, an dem das Modell gespeichert werden soll
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Prüfen, ob bereits ein Modell existiert, das weiter trainiert werden kann, ansonsten starte das Training ab Epoche 0
    checkpoint_path = os.path.join(save_dir, 'best_model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        train_losses = checkpoint['train_losses']
    else:
        start_epoch = 0
        best_loss = float('inf')
        train_losses = []

    results = {"train_loss": [], "test_loss": []}

    best_loss = float('inf')

    for epoch in tqdm(range(epochs)):

        # Aufrufen der Funktionen train_step und test_step
        train_loss, train_ssim = train_step(model=model,
                                            dataloader=train_dataloader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer,
                                            epoch=epoch)

        test_loss, test_ssim = test_step(model=model,
                                         dataloader=test_dataloader,
                                         loss_fn=loss_fn,
                                         epoch=epoch)
        # Anzeigen des Trainingsfortschritts alle 5 Epochen
        if epoch % 5 == 0:
            print(f"Epoche: {epoch+1} | "
                  f"Trainingsverlust: {train_loss:.4f} | "
                  f"Testverlust: {test_loss:.4f} | "
                  f"Trainings-SSIM: {train_ssim:.2f}% | "
                  f"Test-SSIM: {test_ssim:.2f}% | ")

        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)

        # Überprüfen, ob der aktuelle Trainingsverlust der beste Trainingsverlust ist
        if train_loss < best_loss:
            best_loss = train_loss
            # Speichern des Modells
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'train_losses': results["train_loss"]
            }
            torch.save(checkpoint, checkpoint_path)

    # Checkpoint erstellen, um das Training fortzusetzen, wenn es unterbrochen wird, oder um das beste Modell zu laden, das während des Trainings trainiert wurde.
    checkpoint = {
        'epoch': epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
        'train_losses': results["train_loss"]
    }
    torch.save(checkpoint, checkpoint_path)

    return results

#train(model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs=NUM_EPOCHS)

model_results = train(model=model,
                      train_dataloader=train_dataloader,
                      test_dataloader=test_dataloader,
                      optimizer=optimizer,
                      loss_fn=criterion,
                      epochs=NUM_EPOCHS)


def load_model(model_name):
    # Diese Funktion lädt ein vortrainiertes Modell aus einer Datei

    # Der Model-Path beschreibt den Pfad zu dem Ordner, in dem das Model gespeichert ist
    model_path = Path("models")
    # Der Model-Save-Path beschreibt den vollständigen Pfad zur Datei
    model_save_path = model_path / model_name

    # Das Loaded-Model lädt das vortrainierte Modell
    loaded_model = Autoencoder()

    # Die Checkpoint-Informationen, die beim Speichern übergeben wurden, werden geladen
    checkpoint = torch.load(model_save_path)

    # Die Methode `loaded_model.load_state_dict()` lädt den Zustand des vorher trainierten Modells in das neue Modell.
    loaded_model.load_state_dict(checkpoint['model_state_dict'])

    # Die Methode `loaded_model.to()` verschiebt das Modell auf die GPU, falls verfügbar.
    loaded_model = loaded_model.to(device)

    # Das return-Statement gibt das vollständig geladene Modell zurück
    return loaded_model


# Den Pfad zu dem vortrainierten Modell angeben
model_name = '/..../best_model.pth'#Pfad zu dem vortrainierten Modell angeben
# Die Funktion load_model aufrufen und das Modell laden
model_loaded = load_model(model_name)


def image_reconstructed_plot(model: torch.nn.Module,
                             image_path: str,
                             save_path: str,
                             transform=None,
                             device: torch.device = device):

    # Die Funktion image_reconstructed_plot() plottet ein Bild vor und nach dem Durchlaufen des Autoencoders sowie die SSIM-Kontur.
    # Die Funktion nimmt als Eingabe ein Modell, einen Bildpfad, einen Pfad zum Speichern des Plots und einen optionalen Transformator entgegen.
    # Die Funktion plottet das Originalbild, das rekonstruierte Bild und die SSIM-Kontur.
    # Die SSIM-Kontur ist ein Bild, das die Ähnlichkeit zwischen dem Originalbild und dem rekonstruierten Bild darstellt.

    fig, ax = plt.subplots(1, 3)
    plt.gray()

    img = Image.open(image_path)

    if transform:
        img_transformed = transform(img)

    model.to(device)
    model.eval()
    with torch.no_grad():
        input_image = img_transformed.unsqueeze(dim=0).to(device)
        reconstructed = model(input_image)

    original = img_transformed.permute(1, 2, 0).cpu()

    defect = image_path.split("/")
    # Anzeigen des Originalbildes
    ax[0].imshow(original)
    ax[0].set_title('Original')
    ax[0].axis("off")
    # Anzeigen des rekonstruierten Bildes
    reconstructed = reconstructed.squeeze(0).permute(1, 2, 0).cpu()
    ax[1].imshow(reconstructed)
    ax[1].set_title("Rekonstruiert")
    ax[1].axis("off")

    img_old = np.array(original.squeeze(2))
    img_new = np.array(reconstructed.squeeze(2))

    pixel_range = max(np.max(img_new), np.max(img_old)) - min(np.min(img_new), np.min(img_old))
    _, S = ski_ssim(img_old, img_new, full=True, channel_axis=False, data_range=1)  # data range set to 1 or pixel_range, depending on desired precision
    # Anzeigen der SSIM-Kontur
    ax[2].imshow(1-S, vmax=1, cmap='jet')
    ax[2].set_title("SSIM")
    ax[2].axis("off")

    plt.savefig(save_path)
    plt.close(fig)

save_dir = '/Path_to_safe_dir' # Pfad anpassen, an dem das Modell gespeichert werden soll

train_image_path = ('/Pfad/zum/Trainingsbild')
test_image_path_1 = ('/Pfad/zum/Testbild_1')
test_image_path_2 = ('/Pfad/zum/Testbild_2')

image_reconstructed_plot(model=model,
                         image_path=train_image_path,
                         save_path=os.path.join(save_dir, 'train_image.png'),
                         transform=data_transforms_standard,
                         device=device)

image_reconstructed_plot(model=model,
                         image_path=test_image_path_1,
                         save_path=os.path.join(save_dir, 'test_image_1.png'),
                         transform=data_transforms_standard,
                         device=device)

image_reconstructed_plot(model=model,
                         image_path=test_image_path_2,
                         save_path=os.path.join(save_dir, 'test_image_2.png'),
                         transform=data_transforms_standard,
                         device=device)
