import shutil
import os
from datetime import datetime

def copy_directory(src, dest, exclude_dirs):
    exclude_dirs = [os.path.join(src, ex) for ex in exclude_dirs]
    
    for root, dirs, files in os.walk(src):
        # Construct the destination path
        dest_path = root.replace(src, dest, 1)
        
        # Skip excluded directories
        dirs[:] = [d for d in dirs if os.path.join(root, d) not in exclude_dirs]
        
        # Create the destination directory if it doesn't exist
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        
        # Copy files
        for file in files:
            src_file = os.path.join(root, file)
            dest_file = os.path.join(dest_path, file)
            shutil.copy2(src_file, dest_file)


exclude_dirs = [r".git", r".github", r".vscode", r"__pycache__"]

# def zip_directory():

directory_path = os.path.join(os.getcwd(), r'seat')

if os.path.exists(directory_path):
    package_path = os.path.join(directory_path, r"seat_qgis_plugin", r"seat_qgis_plugin")
    print(r"Creating Package")

    if os.path.exists(os.path.join(directory_path, package_path)):
        shutil.rmtree(package_path)
        
    copy_directory(directory_path, package_path, exclude_dirs)        
    # shutil.copytree(directory_path, package_path)
    
    current_date = datetime.now().strftime('%Y%m%d')
    output_filename = f'seat_qgis_plugin_{current_date}'

    # Create a zip archive of the directory
    shutil.make_archive(output_filename, 'zip', os.path.dirname(package_path))
    print(f'Zip archive {output_filename} created successfully')
    shutil.rmtree(package_path)
else:
    print("Directory ./seat not found. Validate current directory")

# Example usage
# zip_directory()