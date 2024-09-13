"""
Creates zip package for QGIS plugin import
"""
import shutil
import os
from datetime import datetime


def copy_directory(src, dest, dirs_to_exclude):
    """Copies seat directory

    :param src: source directory
    :type src: str
    :param dest: destination directory
    :type dest: str
    :param dirs_to_exclude: list
    :type dirs_to_exclude: directories to exclude form copy
    """
    dirs_to_exclude = [os.path.join(src, ex) for ex in dirs_to_exclude]

    for root, dirs, files in os.walk(src):
        # Construct the destination path
        dest_path = root.replace(src, dest, 1)

        # Skip excluded directories
        dirs[:] = [d for d in dirs if os.path.join(root, d) not in dirs_to_exclude]

        # Create the destination directory if it doesn't exist
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)

        # Copy files
        for file in files:
            src_file = os.path.join(root, file)
            dest_file = os.path.join(dest_path, file)
            shutil.copy2(src_file, dest_file)


exclude_dirs = [r".git", r".github", r".vscode", r"__pycache__"]

directory_path = os.path.join(os.getcwd(), r"seat")

if os.path.exists(directory_path):
    package_path = os.path.join(directory_path, r"seat_qgis_plugin", r"seat")
    print(r"Creating Package")

    if os.path.exists(os.path.dirname(package_path)):
        shutil.rmtree(os.path.dirname(package_path))

    copy_directory(directory_path, package_path, exclude_dirs)

    current_date = datetime.now().strftime("%Y%m%d")
    output_filename = f"seat_qgis_plugin_{current_date}"

    # Create a zip archive of the directory
    shutil.make_archive(output_filename, "zip", os.path.dirname(package_path))
    print(f"Zip archive {output_filename} created successfully")
    shutil.rmtree(os.path.dirname(package_path))
else:
    print("Directory ./seat not found. Validate current directory")
