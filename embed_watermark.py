import os
import cv2
import numpy as np
from scipy.fftpack import dct, idct

# ------------------------------
# Helper Functions
# ------------------------------

def apply_dct(block):
    """ Apply Discrete Cosine Transform (DCT) to an 8x8 block """
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def apply_idct(block):
    """ Apply Inverse Discrete Cosine Transform (IDCT) to an 8x8 block """
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

# ------------------------------
# Watermark Embedding Function (Color)
# ------------------------------

def embed_watermark_color(image, watermark, alpha=15):
    """
    Embed a watermark in all three color channels (R, G, B) separately.

    Parameters:
        image (numpy.ndarray): Original color image (BGR format).
        watermark (numpy.ndarray): Watermark image (resized to fit).
        alpha (float): Embedding strength factor.

    Returns:
        watermarked_image (numpy.ndarray): Image with embedded watermark.
    """
    h, w, _ = image.shape

    # Ensure image dimensions are multiples of 8
    h_new = (h // 8) * 8
    w_new = (w // 8) * 8
    image = cv2.resize(image, (w_new, h_new))  # Resize the image safely

    # Resize watermark to fit within the DCT domain of the image
    wm_resized = cv2.resize(watermark, (w_new // 8, h_new // 8))

    # Convert watermark to binary (0 or 1)
    _, wm_binary = cv2.threshold(wm_resized, 127, 1, cv2.THRESH_BINARY)

    # Split image into color channels
    b, g, r = cv2.split(image)

    # Process each channel separately
    b_wm = embed_watermark(b, wm_binary, alpha)
    g_wm = embed_watermark(g, wm_binary, alpha)
    r_wm = embed_watermark(r, wm_binary, alpha)

    # Merge the watermarked color channels
    watermarked_image = cv2.merge([b_wm, g_wm, r_wm])

    return watermarked_image

def embed_watermark(channel, watermark, alpha):
    """
    Embed the watermark into a single color channel using DCT.

    Parameters:
        channel (numpy.ndarray): A single color channel (grayscale).
        watermark (numpy.ndarray): Binary watermark.
        alpha (float): Embedding strength factor.

    Returns:
        watermarked_channel (numpy.ndarray): The watermarked color channel.
    """
    h, w = channel.shape
    watermarked_channel = np.zeros_like(channel, dtype=np.float32)

    # Process the image in 8x8 blocks
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = channel[i:i+8, j:j+8].astype(np.float32)
            dct_block = apply_dct(block)

            # Ensure we do not go out of bounds
            wm_x = min(i // 8, watermark.shape[0] - 1)
            wm_y = min(j // 8, watermark.shape[1] - 1)
            wm_bit = watermark[wm_x, wm_y]

            # Embed watermark in mid-frequency coefficient
            dct_block[4, 3] += alpha * wm_bit  # Modify a mid-frequency coefficient

            # Apply inverse DCT
            watermarked_channel[i:i+8, j:j+8] = apply_idct(dct_block)

    # Normalize and convert back to uint8
    watermarked_channel = np.clip(watermarked_channel, 0, 255).astype(np.uint8)

    return watermarked_channel

# ------------------------------
# Process All Color Images in Folder
# ------------------------------

def process_images(input_folder, output_folder, watermark_path):
    """
    Process all color images in the input folder and embed the watermark.

    Parameters:
        input_folder (str): Path to the folder containing original images.
        output_folder (str): Path to save watermarked images.
        watermark_path (str): Path to the watermark image.
    """
    # Load the watermark image
    watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
    if watermark is None:
        print("Error: Could not load watermark image!")
        return

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Process each image in the folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # Supports JPG, JPEG, PNG
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Load the color image
            image = cv2.imread(input_path)
            if image is None:
                print(f"Skipping {filename}: Unable to read.")
                continue

            print(f"Processing {filename}...")

            # Embed watermark in the color image
            watermarked_image = embed_watermark_color(image, watermark)

            # Save the watermarked image (high quality for JPEG)
            cv2.imwrite(output_path, watermarked_image, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print("Watermarking process completed!")

# ------------------------------
# Main Execution
# ------------------------------

if __name__ == "__main__":
    input_folder = "./test_directory/2048x2048"         # Folder containing original images (JPG, PNG)
    output_folder = "./res(watermarked)/2048x2048"        # Folder to store watermarked images
    watermark_path = "./watermarks/watermark2048.png"  # Path to the watermark image

    process_images(input_folder, output_folder, watermark_path)

