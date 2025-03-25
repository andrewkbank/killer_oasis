import tkinter as tk
import os
from PIL import Image

# Get screen resolution
root = tk.Tk()
width = root.winfo_screenwidth()
height = root.winfo_screenheight()
root.destroy()

print(f"Screen resolution: {width}x{height}")

# Paths
input_folder = "./screenshots"
output_folder = "./cropped_output"
os.makedirs(output_folder, exist_ok=True)

def count_hearts(img):
    """
    Counts the number of hearts in the healthbar of img
    Conveniently detects the deathscreen since it contains no hearts
    Known bugs:
        - some slight x-shifts causes a full health bar to be 16 hp instead of 20 in some aspect ratios
        - discoloring of the healthbar (ie: whither affect or poisoning) might be evaluated to 0 hp
    None of those bugs should be too bad. They should results in false positives (instead of false negatives).
    """
    width, height = img.size
    
    # Compute scale factor based on native UI resolution (480x270)
    NATIVE_WIDTH = 480
    NATIVE_HEIGHT = 270
    scale_factor = height / NATIVE_HEIGHT

    # Health bar position in Minecraft's native resolution
    health_bar_x_start = int((width // 2) - 91 * scale_factor)
    health_bar_y = int(height - 32 * scale_factor)

    health_bar_width = int(81 * scale_factor)
    health_bar_height = int(9 * scale_factor)

    # Crop and save health bar
    health_bar_region = img.crop((health_bar_x_start, health_bar_y - health_bar_height, 
                                  health_bar_x_start + health_bar_width, health_bar_y))
    #health_bar_region.save(output_path)

    # Count hearts by sampling 20 pixels across the width
    pixels = health_bar_region.load()
    y_sample = int(4 * health_bar_height / 9)  # 4/9 from the top

    heart_count = 0
    for i in range(20):
        x_sample = int(i * health_bar_width / 20 + health_bar_width/40)
        r, g, b, *a = pixels[x_sample, y_sample]  # Extract RGBA or RGB values

        # Check if pixel is red (Minecraft hearts are red)
        if r > 150 and g < 100 and b < 100:  # Adjust threshold if necessary
            heart_count += 1

    return heart_count

# Process all images
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        input_path = os.path.join(input_folder, filename)
        #output_path = os.path.join(output_folder, f"cropped_{filename}")
        img = Image.open(input_path)
        heart_count = count_hearts(img)
        print(f"Detected {heart_count} half-hearts in {input_path}")

print("Health bar extraction and analysis complete.")
