import os
import utils.paracousti_fxns as paracousti_fxns
import xarray as xr
import matplotlib.pyplot as plt

paracousti_directory = r"./paracosti"  # update to directory of paracousti files
save_directory = r"./output"  # update to directory to save paracousti files
os.makedirs(save_directory, exist_ok=True)
weights = "All"
status = paracousti_fxns.calc_paracousti_metrics(
    paracousti_directory, save_directory, weights="All"
)
