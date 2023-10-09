import os
from PIL import Image

# Directory containing the images
image_directory = '.'

# Desired maximum file size in kB
max_file_size_kb = 0.1


def compress_image(image_path, output_path, quality=100):
    """
    Compress the image and save it to output_path.
    The quality parameter controls the quality of the compression.
    Lowering the quality value will decrease the file size.
    """
    image = Image.open(image_path).convert(
        'RGB')  # Ensure the image is in RGB mode
    max_width = 1000
    width_percent = (max_width / float(image.size[0]))
    height_size = int((float(image.size[1]) * float(width_percent)))
    resized_img = image.resize((max_width, height_size), Image.ANTIALIAS)

    resized_img.save(output_path, 'WEBP', quality=quality)


def main():
    for filename in os.listdir(image_directory):
        if filename.endswith('.webp'):
            print(f"Processing {filename}...")
            file_path = os.path.join(image_directory, filename)
            output_path = os.path.splitext(file_path)[0] + '.webp'

            # Get initial file size in kB
            initial_file_size_kb = os.path.getsize(file_path) / 1024
            print(f"{filename}: Initial Size: {initial_file_size_kb:.2f} kB")

            if initial_file_size_kb > max_file_size_kb:
                compress_image(file_path, output_path)

                # Get final file size in kB
                final_file_size_kb = os.path.getsize(output_path) / 1024
                print(
                    f"{os.path.basename(output_path)}: Final Size: {final_file_size_kb:.2f} kB\n")


if __name__ == "__main__":
    main()
