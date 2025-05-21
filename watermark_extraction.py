import os
import cv2
import numpy as np
from skimage.restoration import denoise_tv_chambolle  # For uniform noise
from skimage.restoration import denoise_nl_means  # For Poisson noise

def extract_watermark(image, block_size=8, watermark_pos=(4,3)):
    """
    Extracts a watermark by computing the DCT on each block and retrieving the coefficient at watermark_pos.
    
    Args:
        image: Grayscale image (numpy array).
        block_size: Size of the block (default 8).
        watermark_pos: The (row, col) of the DCT coefficient where the watermark was embedded.
        
    Returns:
        watermark: Extracted watermark image (numpy array).
    """
    h, w = image.shape
    blocks_y, blocks_x = h // block_size, w // block_size
    watermark = np.zeros((blocks_y, blocks_x), dtype=np.float32)
    
    for i in range(blocks_y):
        for j in range(blocks_x):
            block = image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            dct_block = cv2.dct(np.float32(block))
            watermark[i, j] = dct_block[watermark_pos]

    # Normalize the watermark to [0,255]
    watermark_norm = cv2.normalize(watermark, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(watermark_norm)

def apply_denoising_filter(watermark, noise_type):
    """
    Applies a denoising filter based on noise type.
    
    Args:
        watermark: Extracted watermark image.
        noise_type: Type of noise in the image.
    
    Returns:
        watermark_denoised: Filtered watermark image.
    """
    noise_type = noise_type.lower()
    if "gaussian" in noise_type:
        watermark_denoised = cv2.GaussianBlur(watermark, (3, 3), 0)
    elif "salt" in noise_type or "pepper" in noise_type:
        watermark_denoised = cv2.medianBlur(watermark, 3)
    elif "speckle" in noise_type:
        watermark_denoised = cv2.bilateralFilter(watermark, d=5, sigmaColor=75, sigmaSpace=75)
    elif "poisson" in noise_type:
        watermark_denoised = denoise_nl_means(watermark, h=10, fast_mode=True)
        watermark_denoised = np.uint8(watermark_denoised * 255)  # Normalize to 0-255
    elif "uniform" in noise_type:
        watermark_denoised = denoise_tv_chambolle(watermark, weight=0.1)
        watermark_denoised = np.uint8(watermark_denoised * 255)  # Normalize to 0-255
    else:
        watermark_denoised = watermark.copy()
    
    return watermark_denoised

def process_noisy_watermarks(original_dataset, noisy_images, output_folder, block_size=8, watermark_pos=(4,3)):
    """
    Processes the noisy images to extract watermarks using inverse DCT, applies denoising filters,
    and saves the extracted watermarks in a folder structure mirroring the noisy images folder.
    
    Args:
        original_dataset: Path to the original images folder.
        noisy_images: Path to the noisy images folder.
        output_folder: Path to save extracted watermarks.
        block_size: DCT block size.
        watermark_pos: Position of the watermark in DCT coefficients.
    """
    for resolution in os.listdir(noisy_images):
        res_path = os.path.join(noisy_images, resolution)
        if not os.path.isdir(res_path):
            continue

        for noise_type in os.listdir(res_path):
            noise_path = os.path.join(res_path, noise_type)
            if not os.path.isdir(noise_path):
                continue

            out_dir = os.path.join(output_folder, resolution, noise_type)
            os.makedirs(out_dir, exist_ok=True)

            for file in os.listdir(noise_path):
                if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    continue

                noisy_img_path = os.path.join(noise_path, file)
                noisy_img = cv2.imread(noisy_img_path, cv2.IMREAD_COLOR)
                if noisy_img is None:
                    print(f"Skipping {file}: unable to load image.")
                    continue

                gray = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2GRAY)
                watermark_extracted = extract_watermark(gray, block_size, watermark_pos)
                watermark_denoised = apply_denoising_filter(watermark_extracted, noise_type)

                out_path = os.path.join(out_dir, file)
                cv2.imwrite(out_path, watermark_denoised)
                print(f"Saved extracted watermark: {out_path}")

if __name__ == "__main__":
    original_dataset = r"./testing/test_directory"
    noisy_images = r"./testing/res_with_noise"
    output_folder = r"./testing/noisy_watermarks"
    
    process_noisy_watermarks(original_dataset, noisy_images, output_folder)
