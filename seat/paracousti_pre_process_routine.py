# Processing of parAcousti files can be resource instensive depending on the grid resolution and weights selected. This may take several hours to complete.
# NOTES:
# Requires installation of xarray with netCDF4 (https://docs.xarray.dev/en/stable/getting-started-guide/installing.html)
#     python -m pip install "xarray[io]"
# Requires installation of scipy (https://scipy.org/install/)
#     This will be included with "xarray[io]" or python -m pip install scipy
# Requires intallation of tqdm (https://tqdm.github.io/)
#     python -m pip install tqdm

import os
import utils.paracousti_fxns as paracousti_fxns
import xarray as xr
import matplotlib.pyplot as plt

paracousti_directory = r"C:\Users\aellenson\OneDrive - Integral Consulting Inc\C1308 Sandia\Tutorial_Files\Development\DEMO Files Tabbed GUI - Acoustics\pacwave\paracousti_files\\"
save_directory = r"C:\Users\aellenson\OneDrive - Integral Consulting Inc\C1308 Sandia\Tutorial_Files\Development\DEMO Files Tabbed GUI - Acoustics\pacwave\paracousti_files_with_metrics_TU\\"
os.makedirs(save_directory, exist_ok=True)
weights = "All"
status = paracousti_fxns.calc_paracousti_metrics(
    paracousti_directory, save_directory, weights="TU"
)
