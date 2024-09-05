import os
import utils.paracousti_fxns as paracousti_fxns
import xarray as xr
import matplotlib.pyplot as plt

paracousti_path = r"./paracosti"  # update to directory of paracousti files
save_path = r"./output"  # update to directory to save paracousti files
os.makedirs(save_path, exist_ok=True)
weights = "All"
status = paracousti_fxns.calc_paracousti_metrics(
    paracousti_path, save_path, weights="All"
)
