from PIL import Image
import os

def squarify_image(img: Image.Image) -> Image.Image:
    target_size = max(img.width, img.height)   
    # Create a new image with the same size and white background
    square_image = Image.new("L", (target_size, target_size), 255)
    square_image.paste(img, ((target_size - img.width) // 2, (target_size - img.height) // 2))
    return square_image

input_folder = os.path.join(os.getcwd(), "shoe-model\\train")
output_folder = os.path.join(os.getcwd(), "silhouette-model\\train")

print("Input folder: " + input_folder)
print("Output folder: " + output_folder)

for subfolder in os.listdir(input_folder):
    print("Looping through " + subfolder)
    subfolder_dir = os.path.join(input_folder, subfolder)
    output_subfolder_dir = os.path.join(output_folder, subfolder)
    if not os.path.exists(output_subfolder_dir):
        os.makedirs(output_subfolder_dir)
    for filename in os.listdir(subfolder_dir):
        print(filename)
        if filename.endswith('.webp'):
            webp_image = Image.open(os.path.join(subfolder_dir, filename)).convert("RGBA")
            output_filename = filename.split('.')[0] + '.jpg'
            
            # Create a new image with the same size and white background
            white_background = Image.new('RGBA', webp_image.size, (255, 255, 255, 255))
            white_background.paste(webp_image, mask=webp_image.split()[3])  # Paste the webp image using its alpha channel as a mask
            rgb_image = white_background.convert('RGB')
        
            # Convert the image to grayscale
            grayscale_image = rgb_image.convert('L')
            
            squared_image = squarify_image(grayscale_image)
            
            squared_image.save(os.path.join(output_subfolder_dir, output_filename), 'JPEG')
