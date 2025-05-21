import os
import numpy as np
from PIL import Image

# Noise functions
def add_gaussian_noise(image_array, mean=0, sigma=25):
    """Adds Gaussian noise to the image."""
    gaussian_noise = np.random.normal(mean, sigma, image_array.shape)
    noisy_image = image_array + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

def add_salt_and_pepper_noise(image_array, salt_prob=0.05, pepper_prob=0.05):
    """Adds Salt-and-Pepper noise to the image."""
    noisy_image = np.copy(image_array)
    # Create a random matrix for noise
    rand_matrix = np.random.rand(*image_array.shape[:2])
    # Apply salt noise (white pixels)
    noisy_image[rand_matrix < salt_prob] = 255
    # Apply pepper noise (black pixels)
    noisy_image[rand_matrix > 1 - pepper_prob] = 0
    return noisy_image

def add_speckle_noise(image_array):
    """Adds Speckle noise to the image."""
    noise = np.random.randn(*image_array.shape)
    noisy_image = image_array + image_array * noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

def add_poisson_noise(image_array):
    """Adds Poisson noise to the image safely."""
    image_array = image_array.astype(np.float32)  # Convert to float
    noisy_image = np.random.poisson(image_array)  # Apply Poisson noise
    noisy_image = np.clip(noisy_image, 0, 255)  # Ensure valid range
    return noisy_image.astype(np.uint8)  # Convert back to uint8

def add_uniform_noise(image_array, low=-50, high=50):
    """Adds Uniform noise to the image."""
    uniform_noise = np.random.uniform(low, high, image_array.shape)
    noisy_image = image_array + uniform_noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

# Define the noise functions in a dictionary for easy iteration.
noise_functions = {
    "Gaussian": add_gaussian_noise,
    "SaltPepper": add_salt_and_pepper_noise,
    "Speckle": add_speckle_noise,
    "Poisson": add_poisson_noise,
    "Uniform": add_uniform_noise
}

# Define directories
INPUT_FOLDER = "./testing/test_directory/2048x2048"  # Folder containing your watermarked images
OUTPUT_BASE_FOLDER = "./testing/res(with_noise)/2048x2048"   # Base folder to save noisy images

# Create output directories for each noise type
for noise_name in noise_functions.keys():
    output_dir = os.path.join(OUTPUT_BASE_FOLDER, noise_name)
    os.makedirs(output_dir, exist_ok=True)

# Process each image in the input folder
for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        input_path = os.path.join(INPUT_FOLDER, filename)
        try:
            with Image.open(input_path) as img:
                # Ensure image is in RGB format
                img = img.convert("RGB")
                image_array = np.array(img)
        except Exception as e:
            print(f"⚠ Error opening {filename}: {e}")
            continue

        base_name, _ = os.path.splitext(filename)
        # Apply each noise function and save the resulting image
        for noise_name, noise_func in noise_functions.items():
            try:
                noisy_array = noise_func(image_array)
                noisy_img = Image.fromarray(noisy_array)
                # Construct filename like image1_Gaussian.png
                output_filename = f"{base_name}_{noise_name}.png"
                output_path = os.path.join(OUTPUT_BASE_FOLDER, noise_name, output_filename)
                noisy_img.save(output_path)
                print(f"✔ Saved: {output_path}")
            except Exception as e:
                print(f"⚠ Error processing {filename} with {noise_name} noise: {e}")

print(" All images processed and saved!")
