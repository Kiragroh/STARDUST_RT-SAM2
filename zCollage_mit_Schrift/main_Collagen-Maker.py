import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw, ImageFont, ImageOps
import random
from datetime import datetime

# Keywords to exclude from filenames
EXCLUDE_KEYWORDS = ['kopf', 'mr', 'km', 'met', 'mb', 'stx', 'hirn', 'bein', 'hno', 'schädel', 'hals']

# Configuration settings
COLLAGE_SETTINGS = {
    'images_per_collage': 420,  # Number of images per collage
    'aspect_ratio_mode': 'zoom_out',  # 'zoom_in' or 'zoom_out'
    'crop_factor': 1,  # Percentage of original image to keep (0.0 to 1.0)
    'tint_strength': 0.5  # Global tint strength control
}

def enhance_red(image):
    """Enhance red colors in an image using HSV color space"""
    # Convert PIL Image to OpenCV format
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define red ranges
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    # Dilatiere die Maske, um die rote Linie dicker erscheinen zu lassen
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))  # Anpassbar, z.B. (5,5) für noch mehr Dicke
    mask_dilated = cv2.dilate(mask, kernel, iterations=1)

    # Berechne die "neuen" Pixel, die in der dilatierten Maske liegen, aber nicht in der Originalmaske
    new_pixels_mask = cv2.subtract(mask_dilated, mask)

    # Erstelle eine Kopie des HSV-Bildes, um gezielt zu bearbeiten
    hsv_modified = hsv.copy()

    # Für die neu hinzugefügten Pixel: Setze den Farbton (Hue) auf einen festen Rot-Wert
    # (in OpenCV's HSV liegt Rot z.B. ungefähr bei 0 bis 10; hier wählen wir 5 als Beispiel)
    desired_red_hue = 5
    hsv_modified[new_pixels_mask > 0, 0] = desired_red_hue
    # Optional: Du kannst auch Sättigung und Helligkeit für diese Pixel auf hohe Werte setzen, 
    # um einen kräftigen Rotton zu erzwingen:
    hsv_modified[new_pixels_mask > 0, 1] = 255
    hsv_modified[new_pixels_mask > 0, 2] = 255

    # Optional: Falls du auch die bereits roten Pixel weiter verstärken möchtest,
    # kannst du diesen Schritt auch auf alle Pixel in mask_dilated anwenden:
    # hsv_modified[mask_dilated > 0, 0] = desired_red_hue
    # hsv_modified[mask_dilated > 0, 1] = np.clip(hsv_modified[mask_dilated > 0, 1] * 1.5, 0, 255)
    # hsv_modified[mask_dilated > 0, 2] = np.clip(hsv_modified[mask_dilated > 0, 2] * 1.5, 0, 255)

        
    # Convert back to BGR then to PIL Image
    result = cv2.cvtColor(hsv_modified, cv2.COLOR_HSV2BGR)
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

def adjust_image(image_path):
    """Adjust and enhance an image"""
    image = Image.open(image_path)
    image = enhance_red(image)  # Enhance reds first
    image = image.resize((256, 256))
    
    # Convert to grayscale if needed
    image = image.convert("RGB")
    
    # Handle aspect ratio based on mode
    width, height = image.size
    if COLLAGE_SETTINGS['aspect_ratio_mode'] == 'zoom_out':
        # Zoom out: fit entire image, add padding if needed
        ratio = width / height
        target_ratio = 16 / 9
        if ratio > target_ratio:
            # Image is wider than 16:9
            new_height = int(width / target_ratio)
            padding = (new_height - height) // 2
            image = ImageOps.expand(image, (0, padding, 0, padding), fill='black')
        else:
            # Image is taller than 16:9
            new_width = int(height * target_ratio)
            padding = (new_width - width) // 2
            image = ImageOps.expand(image, (padding, 0, padding, 0), fill='black')
    else:
        # Zoom in: crop to 16:9
        ratio = width / height
        target_ratio = 16 / 9
        if ratio > target_ratio:
            # Image is wider than 16:9
            new_width = int(height * target_ratio)
            left = (width - new_width) // 2
            image = image.crop((left, 0, left + new_width, height))
        else:
            # Image is taller than 16:9
            new_height = int(width / target_ratio)
            top = (height - new_height) // 2
            image = image.crop((0, top, width, top + new_height))
    
    # Apply cropping factor
    width, height = image.size
    crop_size = int(min(width, height) * COLLAGE_SETTINGS['crop_factor'])
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    image = image.crop((left, top, right, bottom))
    
    # Enhance brightness and contrast
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.2)
    
    return image

def filter_images(image_folder):
    # Get all PNG files excluding those with specific keywords
    images = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
              if f.lower().endswith('.png') and not any(kw in f.lower() for kw in EXCLUDE_KEYWORDS)]
    
    if len(images) == 0:
        print("No images found! Please check if:")
        print("1. The folder contains PNG files")
        print(f"2. The files don't contain these keywords: {EXCLUDE_KEYWORDS}")
        return None
    
    print(f"Found {len(images)} suitable images for collage")
    return images

def create_variant(image):
    # Create different variants of the image
    variant = image.copy()
    # Apply random transformations
    brightness_factor = random.uniform(0.8, 1.2)
    contrast_factor = random.uniform(0.8, 1.2)
    enhancer = ImageEnhance.Brightness(variant)
    variant = enhancer.enhance(brightness_factor)
    enhancer = ImageEnhance.Contrast(variant)
    variant = enhancer.enhance(contrast_factor)
    return variant

def apply_tint(image, color_index):
    # Apply rotating tint colors
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]  # Blue, Red, Green
    tint = Image.new('RGB', image.size, colors[color_index % len(colors)])
    return Image.blend(image, tint, COLLAGE_SETTINGS['tint_strength'])

def create_collage(images, save_path):
    # Define letter patterns
    letter_patterns = {
        "S": [
            "11111",
            "10000",
            "11111",
            "00001",
            "11111"
        ],
        "T": [
            "11111",
            "00100",
            "00100",
            "00100",
            "00100"
        ],
        "A": [
            "01110",
            "10001",
            "11111",
            "10001",
            "10001"
        ],
        "R": [
            "11110",
            "10001",
            "11110",
            "10100",
            "10010"
        ],
        "D": [
            "11110",
            "10001",
            "10001",
            "10001",
            "11110"
        ],
        "U": [
            "10001",
            "10001",
            "10001",
            "10001",
            "11111"
        ]
    }
    
    # Create 5 variants for each collage
    for variant_num in range(5):
        # 16:9 aspect ratio collage
        cols = 42
        rows = 7
        
        # Create blank canvas
        image_size = 256
        collage_width = cols * image_size
        collage_height = rows * image_size
        collage = Image.new("RGB", (collage_width, collage_height), (255, 255, 255))
        
        # Randomly select images for this collage
        selected_images = random.sample(images, min(COLLAGE_SETTINGS['images_per_collage'], len(images)))
        
        # Fill grid with unedited images
        for row in range(rows):
            for col in range(cols):
                idx = row * cols + col
                if idx < len(selected_images):
                    img = selected_images[idx]
                    collage.paste(img, (col * image_size, row * image_size))
        
        # Overlay STARDUST pattern
        word = "STARDUST"
        text_width = len(word) * 5
        start_col = (cols - text_width) // 2
        start_row = (rows - 5) // 2
        
        current_col = start_col
        color_index = 0
        for letter in word:
            pattern = letter_patterns.get(letter)
            if pattern is None:
                continue
            # Place letter in grid
            for i in range(5):
                for j in range(5):
                    if pattern[i][j] == "1":
                        idx = (start_row + i) * cols + (current_col + j)
                        if idx < len(selected_images):
                            img = selected_images[idx]
                            img = apply_tint(img, color_index)
                            collage.paste(img, ((current_col + j) * image_size, (start_row + i) * image_size))
            current_col += 5
            color_index += 1
        
        # Save with variant number and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        variant_path = save_path.replace(".png", f"_v{variant_num+1}_{timestamp}.png")
        collage.save(variant_path)
        print(f"Saved collage variant: {variant_path}")

def main():
    image_folder = os.path.join(os.getcwd(),"middleSlices")
    output_folder = os.path.join(image_folder, "collages")
    os.makedirs(output_folder, exist_ok=True)
    
    # Get filtered and randomly selected images
    image_paths = filter_images(image_folder)
    
    if not image_paths:
        print("No suitable images found!")
        return
    
    # Process images and create collage
    processed_images = [adjust_image(img) for img in image_paths]
    
    for i in range(5):
        save_path = os.path.join(output_folder, f"collage_{i+1}.png")
        create_collage(processed_images, save_path)

if __name__ == "__main__":
    main()
