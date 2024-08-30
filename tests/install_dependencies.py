import subprocess
import sys

# Default path to the Python executable
default_python_exe = "python3"
# Use the provided path if given, otherwise use the default
python_exe = sys.argv[1] if len(sys.argv) > 1 else default_python_exe

# Upgrade pip
# subprocess.run([python_exe, "-m", "pip", "install", "--upgrade", "pip"], check=True)

# List of packages to install
# packages = ["pandas", "netCDF4", "pytest", "pylint"]

# # Install each package using pip
# for package in packages:
#     subprocess.run([python_exe, "-m", "pip", "install", package], check=True)
