import subprocess

# List of packages to install
packages = ["pandas", "netCDF4"]

# Install each package using pip
for package in packages:
    subprocess.run(["pip", "install", package], check=True)
