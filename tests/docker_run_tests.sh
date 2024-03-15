#!/bin/bash

# Set the name of the Docker container running QGIS
CONTAINER_NAME="qgis-testing-environment"

# Command to execute Python inside the container
PYTHON_CMD="docker exec $CONTAINER_NAME python3"

# Upgrade pip
$PYTHON_CMD -m pip install --upgrade pip

# Check if netCDF4 is installed and install if not
$PYTHON_CMD -m pip show netCDF4 > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "netCDF4 not found, installing..."
    $PYTHON_CMD -m pip install netCDF4
fi

# Run your QGIS-based Python script
docker exec $CONTAINER_NAME python3 /tests/test_shear_stress.py
