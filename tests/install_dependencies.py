import subprocess
import sys
import os

# Default path to the Python executable
default_python_exe = "python3"
# Use the provided path if given, otherwise use the default
python_exe = sys.argv[1] if len(sys.argv) > 1 else default_python_exe

# Path for the virtual environment
venv_path = "/tests_directory/venv"

# Step 1: Create a virtual environment
subprocess.run([python_exe, "-m", "venv", venv_path], check=True)

# Step 2: Use the Python executable from the virtual environment
venv_python_exe = os.path.join(venv_path, "bin", "python")
venv_pip_exe = os.path.join(venv_path, "bin", "pip")

# Step 3: Upgrade pip in the virtual environment
subprocess.run([venv_pip_exe, "install", "--upgrade", "pip"], check=True)

# Step 4: List of packages to install
packages = ["pandas", "netCDF4", "pytest", "rasterio"]

# Step 5: Install each package using pip in the virtual environment
for package in packages:
    subprocess.run([venv_pip_exe, "install", package], check=True)

# Step 6: Output message for successful installation
print("Packages installed successfully in the virtual environment.")
