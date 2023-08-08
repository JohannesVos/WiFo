import os
import random
import shutil
from PIL import Image
from torchvision import transforms

# Sharpening factor
kernel_size = 15

# Define your transformations
transformations = [
    #transforms.GaussianBlur(kernel_size, sigma=(3.0, 3.0)),
    #transforms.ColorJitter(brightness=0.4),
    #transforms.deleteRandomPixels(66)
    transforms.Grayscale(),
    transforms.Resize(256)
    ]


# Specify the input folder path
input_folder = r"C:\Users\johan\Desktop\bottle\test_merged"

# Specify the output folder path for the transformed images
output_folder = r"C:\Users\johan\Desktop\bottle\test_merged_transformed"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get the list of image files in the input folder
image_files = [file for file in os.listdir(input_folder) if file.endswith(('.jpg', '.jpeg', '.png'))]

# Calculate the number of images to transform (x% of the dataset)
num_images = int(len(image_files) *1)

# Randomly select image files for the subset
subset_image_files = random.sample(image_files, num_images)

# Apply the transformation to the subset of images and save them in the output folder
for image_file in subset_image_files:
    # Load the image
    image_path = os.path.join(input_folder, image_file)
    image = Image.open(image_path)
    transformed_image = image

    # Apply the transformations to the image one by one
    for transformation in transformations:
        transformed_image = transformation(transformed_image)

    # Save the transformed image in the output folder
    output_image_path = os.path.join(output_folder, image_file)
    transformed_image.save(output_image_path)

# Copy the remaining images from the input folder to the output folder
for image_file in image_files:
    if image_file not in subset_image_files:
        input_image_path = os.path.join(input_folder, image_file)
        output_image_path = os.path.join(output_folder, image_file)
        shutil.copyfile(input_image_path, output_image_path)


