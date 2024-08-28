"""
/***************************************************************************.

 acoustics_module.py
 Copyright 2023, Integral Consulting Inc. All rights reserved.


 PURPOSE: module for calcualting acoustic signal change from paracousti files

 PROJECT INFORMATION:
 Name: SEAT - Spatial and Environmental Assessment Toolkit
 Number: C1308

 AUTHORS
  Eben Pendelton
  Timothy Nelson (tnelson@integral-corp.com)
  Sam McWilliams (smcwilliams@integral-corp.com)


 NOTES (Data descriptions and any script specific notes)
        1. called by stressor_receptor_calc.py
"""

import os
from scipy.interpolate import griddata
from netCDF4 import Dataset
import pandas as pd
from osgeo import gdal, osr
import numpy as np
from ..utils.stressor_utils import (
    redefine_structured_grid,
    create_raster,
    numpy_array_to_raster,
    calculate_cell_area,
    resample_structured_grid,
    bin_layer,
    secondary_constraint_geotiff_to_numpy,
)


def create_species_array(species_filename, x, y, variable="percent", latlon=False):
    """
    Interpolates or creates an array of percent or density of species

    Parameters
    -----------
    species_filename : str
        File path to species files.
    x : array
        x-coordinate to interpolate onto.
    y : array
        y-coordinate to interpolate onto.
    variable : str, optional
        either 'density' or 'percent' column name for csv files. The default is 'percent'.
    latlon : Bool, optional
        Coordiante Reference System / EPSG code.

    Raises
    ------
    Invalide File Type
        "Invalid File Type. Must be of type .tif or .csv".

    Returns
    -------
    variable_array : array
        array with shape (x).

    """
    # if ((receptor_filename is not None) or (not receptor_filename == "")):
    if not ((species_filename is None) or (species_filename == "")):
        if species_filename.endswith(".tif"):
            data = gdal.Open(species_filename)
            img = data.GetRasterBand(1)
            receptor_array = img.ReadAsArray()
            receptor_array[receptor_array < 0] = 0
            (upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = (
                data.GetGeoTransform()
            )
            cols = data.RasterXSize
            rows = data.RasterYSize
            r_rows = np.arange(rows) * y_size + upper_left_y + (y_size / 2)
            r_cols = np.arange(cols) * x_size + upper_left_x + (x_size / 2)
            if latlon is True:
                r_cols = np.where(r_cols < 0, r_cols + 360, r_cols)
            x_grid, y_grid = np.meshgrid(r_cols, r_rows)
            variable_array = griddata(
                (x_grid.flatten(), y_grid.flatten()),
                receptor_array.flatten(),
                (x, y),
                method="nearest",
                fill_value=0,
            )

        elif species_filename.endswith(".csv"):
            df = pd.read_csv(species_filename)
            columns_keep = ["latitude", "longitude", variable]
            df = df[columns_keep]
            variable_array = griddata(
                (df.longitude.to_numpy(), df.latitude.to_numpy()),
                df[variable].to_numpy(),
                (x, y),
                method="nearest",
                fill_value=0,
            )
        else:
            raise Exception("Invalid File Type. Must be of type .tif or .csv")
    else:
        variable_array = np.zeros(x.shape)
    return variable_array


def find_acoustic_metrics(paracousti_file):
    """Determine all acoustic metrics present in a netCDF (.nc) file.
    
    Parameters
    ----------
    paracousti_file : (str, filepath)
        acoustic netCDF file 
        
    Returns
    -------
    weights : list
        list of weightings in the paracousti_file 
    unweighted_vars : list
       list of unweighted variabless in the paracousti_file 
    weigthed_vars : list
       list of weighted variabless in the paracousti_file 
    """
    ignore_vars = ["octSPL", "XCOR", "YCOR", "ZCOR", "Hw", "Fc", "press_muPa"]
    with Dataset(paracousti_file) as DS:
        avars = list(DS.variables)
        avars = [i for i in avars if i not in set(ignore_vars)]
        weighted_varnames = [i for i in avars if i.endswith(r"_weighted")]
        unweighted_vars = [i for i in avars if i not in set(weighted_varnames)]
        weights = ["None"] + sorted(
            list(set([i.split("_")[0] for i in weighted_varnames]))
        )
        weigthed_vars = sorted(set([i[i.find("_") + 1 :] for i in weighted_varnames]))
    return weights, unweighted_vars, weigthed_vars

def sum_SEL(x):
    """
    function for summing an array of sound exposure levels (SEL) to get cumulative SEL.
    
    Parameters
    ----------
    x :
        list of SEL values (in dB)

    Returns
    -------
        returns total SEL (in dB)
    """
    x_dB = np.asarray(x)
    x_uPa2s = 10 ** (x_dB / 10)
    sum_uPa2s = sum(x_uPa2s)
    sum_dB = 10 * np.log10(sum_uPa2s)
    return sum_dB

def calc_SEL_cum(SEL, duration_seconds):
    """    
    SEL from single second to cumulative
    derived from BOEM 2023 equation 7 for multiple strikes

    Parameters
    ----------
    SEL : array
        single second SEL
    duration_seconds : float
        duration in seconds

    Returns:
    ----------
    array 
        cumulative SEL
    """
    with np.errstate(divide='ignore'):
        cum_sel = SEL + 10 * np.log10(duration_seconds)
    return cum_sel

def calc_probabilistic_metrics(
    paracousti_files,
    conditions_probability,
    threshold,
    Paracousti,
    Baseline,
    XCOR,
    YCOR,
    latlon,
    metric_calc = 'SPL',
    species_folder=None,
    grid_res_species=0
):
    """
    Calculates the probablistic acoutic metric (either SPL or SEL) for a given set of probabilities.
    
    Parameters
    ----------
    paracousti_files : (list | filepath | str)
        list of full filepaths of paracousti files
    conditions_probability : (dataframe)
        dataframe of probabilities for each paracousti_file
    threshold : float
        threshold value (dB)
    Paracousti : array [N, x, y]
        paracousti acoustic metric with axis=0 (N) corresponding to unique files
    Baseline : array [N, x, y]
        baseline acoustic metric with axis=0 (N) corresponding to unique files
    XCOR : array [x,y]
        x-coordinate
    YCOR : array [x,y]
        y-coordinate
    latlon : bool
        True if XCOR,YCOR are lonlat, False if UTM
    metric_calc: str ['SEL'|'SPL'], defaults to 'SPL'
        type of metric, either sound pressure level (SPL) or sound exposure level (SEL)
    species_folder: filepath, str, defaults to None
        filepath of species density/likelihood
    grid_res_species: float, defaults to 0
        grid resolution of species_folder files
        
    Raises
    ------
        FileNotFoundError: "The directory {species_folder} does not exist"

    Returns:
    ----------        
    device: array (x,y)
        probabilistic acoustic metric with devices present
    baseline: array (x,y)
        probabilistic acoustic metric without devices
    stressor: array (x,y)
        probabilistic difference between with and without devices 
    threshold_exceeded: array (x,y)
        % of time threshold exceeded based on probability
    percent_scaled: array (x,y)
        percent of species impacted based on probability
    density_scaled: array (x,y)
        density (number) of species impacted based on probability
    rx: array (x,y)
        x-coordinates of output
    ry: array (x,y)
        y-coordinates of output
    """
    probability = conditions_probability["% of yr"] / 100
    if metric_calc =='SEL':
        seconds_of_day = 24 * 60 * 60 * probability
    # SPL stressor calculations
    for ic, paracousti_file in enumerate(paracousti_files):
        # paracousti files might not have regular grid spacing.
        rx, ry, device_ss = redefine_structured_grid(XCOR, YCOR, Paracousti[ic, :])
        baseline_ss = resample_structured_grid(XCOR, YCOR, Baseline[ic, :], rx, ry)
        if ic == 0:
            device = np.zeros(rx.shape)
            baseline = np.zeros(rx.shape)
            stressor = np.zeros(rx.shape)
            threshold_exceeded = np.zeros(rx.shape)
            percent_scaled = np.zeros(rx.shape)
            density_scaled = np.zeros(rx.shape)
            
        if metric_calc.casefold() == 'SEL'.casefold():
            device_scaled = calc_SEL_cum(device_ss, seconds_of_day.loc[os.path.basename(paracousti_file)])
            baseline_scaled = calc_SEL_cum(baseline_ss, seconds_of_day.loc[os.path.basename(paracousti_file)])
            if ic==0:
                device = device + device_scaled
                baseline = baseline + baseline_scaled
            else:
                device = sum_SEL([device.flatten(), device_scaled.flatten()]).reshape(rx.shape)
                baseline = sum_SEL([baseline.flatten(), baseline_scaled.flatten()]).reshape(rx.shape)
        else: #SPL
            device_scaled = probability.loc[os.path.basename(paracousti_file)] * device_ss
            baseline_scaled = probability.loc[os.path.basename(paracousti_file)] * baseline_ss
            device = device + device_scaled
            baseline = baseline + baseline_scaled
        
        threshold_mask = device_scaled > threshold
        threshold_exceeded[threshold_mask] += probability.loc[os.path.basename(paracousti_file)] * 100

        if not ((species_folder is None) or (species_folder == "")):
            if not os.path.exists(species_folder):
                raise FileNotFoundError(
                    f"The directory {species_folder} does not exist."
                )
            species_percent_filename = conditions_probability.loc[
                os.path.basename(paracousti_file)
            ]["Species Percent Occurance File"]
            species_density_filename = conditions_probability.loc[
                os.path.basename(paracousti_file)
            ]["Species Density File"]
            parray = create_species_array(
                os.path.join(species_folder, species_percent_filename),
                rx,
                ry,
                variable="percent",
                latlon=True,
            )
            darray = create_species_array(
                os.path.join(species_folder, species_density_filename),
                rx,
                ry,
                variable="density",
                latlon=True,
            )
            _, _, square_area = calculate_cell_area(rx, ry, latlon is True)
            # square area of each grid cell
            square_area = np.nanmean(square_area)
            if grid_res_species != 0:
                # ratio of grid cell to species averaged, now prob/density per each grid cell
                ratio = square_area / grid_res_species
            else:
                ratio = 1
            parray_scaled = parray * ratio
            darray_scaled = darray * ratio
            percent_scaled[threshold_mask] += (
                probability.loc[os.path.basename(paracousti_file)] * parray_scaled[threshold_mask]
            )
            density_scaled[threshold_mask] += (
                probability.loc[os.path.basename(paracousti_file)] * darray_scaled[threshold_mask]
            )

    stressor = device - baseline
    return device, baseline, stressor, threshold_exceeded, percent_scaled, density_scaled, rx, ry

def calc_nonprobabilistic_metrics(
    paracousti_files,
    conditions_probability,
    threshold,
    Paracousti,
    Baseline,
    XCOR,
    YCOR,
    latlon,
    metric_calc = 'SPL',
    species_folder=None,
    grid_res_species=0,
    sel_hours=24,
):
    """
    Calculates the nonprobablistic acoutic metric (either SPL or SEL) . For SEL, the calculation is over a fixed time interval for each case.
    
    Parameters
    ----------
    paracousti_files : (list | filepath | str)
        list of full filepaths of paracousti files
    conditions_probability : (dataframe)
        dataframe of probabilities for each paracousti_file
    threshold : float
        threshold value (dB)
    ACOUST_VAR : array [N, x, y]
        paracousti acoustic metric with axis=0 (N) corresponding to unique files
    Baseline : array [N, x, y]
        baseline acoustic metric with axis=0 (N) corresponding to unique files
    XCOR : array [x,y]
        x-coordinate
    YCOR : array [x,y]
        y-coordinate
    latlon : bool
        True if XCOR,YCOR are lonlat, False if UTM
    metric_calc: str ['SEL'|'SPL'], defaults to 'SPL'
        type of metric, either sound pressure level (SPL) or sound exposure level (SEL)
    species_folder: filepath, str, defaults to None
        filepath of species density/likelihood
    grid_res_species: float, defaults to 0
        grid resolution of species_folder files
    sel_hours : scalar float
        number of hour to calculate SEL exposure over (default is 24)
        
    Raises
    ------
        FileNotFoundError: "The directory {species_folder} does not exist"

    Returns:
    ----------        
    device: dict or arrays (x,y)
        key is paracousti_file without extension
        acoustic metric with devices present for each condition
    baseline: dict or arrays (x,y)
        key is paracousti_file without extension
        acoustic metric without devices for each condition
    stressor: dict or arrays (x,y)
        key is paracousti_file without extension
        difference between with and without devices  for each condition
    threshold_exceeded: dict or arrays (x,y)
        key is paracousti_file without extension
        % of time threshold exceeded for each condition
    percent_scaled: dict or arrays (x,y)
        key is paracousti_file without extension
        percent of species impacted for each condition
    density_scaled: dict or arrays (x,y)
        key is paracousti_file without extension
        density (number) of species impacted for each condition
    rx: array (x,y)
        x-coordinates of output
    ry: array (x,y)
        y-coordinates of output
    """

    device = {}
    baseline = {}
    stressor = {}
    threshold_exceeded = {}
    percent_scaled = {}
    density_scaled = {}

    if metric_calc =='SEL':
        duration_seconds = sel_hours * 60 * 60

    for ic, paracousti_file in enumerate(paracousti_files):
        pname = ".".join(os.path.basename(paracousti_file).split('.')[:-1])
        # paracousti files might not have regular grid spacing.
        rx, ry, device_ss = redefine_structured_grid(XCOR, YCOR, Paracousti[ic, :])
        baseline_ss = resample_structured_grid(XCOR, YCOR, Baseline[ic, :], rx, ry)

        if metric_calc.casefold() == 'SEL'.casefold():
            device_scaled = calc_SEL_cum(device_ss, duration_seconds)
            baseline_scaled = calc_SEL_cum(baseline_ss, duration_seconds)
        else: #SPL
            device_scaled = device_ss
            baseline_scaled = baseline_ss
        
        device[pname] = device_scaled
        baseline[pname] = baseline_scaled
        stressor[pname] = device_scaled - baseline_scaled
        threshold_mask = device_scaled > threshold
        threshold_exceeded[pname] = threshold_mask * 100

        if not ((species_folder is None) or (species_folder == "")):
            if not os.path.exists(species_folder):
                raise FileNotFoundError(
                    f"The directory {species_folder} does not exist."
                )
            species_percent_filename = conditions_probability.loc[
                os.path.basename(paracousti_file)
            ]["Species Percent Occurance File"]
            species_density_filename = conditions_probability.loc[
                os.path.basename(paracousti_file)
            ]["Species Density File"]
            parray = create_species_array(
                os.path.join(species_folder, species_percent_filename),
                rx,
                ry,
                variable="percent",
                latlon=True,
            )
            darray = create_species_array(
                os.path.join(species_folder, species_density_filename),
                rx,
                ry,
                variable="density",
                latlon=True,
            )
            _, _, square_area = calculate_cell_area(rx, ry, latlon is True)
            # square area of each grid cell
            square_area = np.nanmean(square_area)
            if grid_res_species != 0:
                # ratio of grid cell to species averaged, now prob/density per each grid cell
                ratio = square_area / grid_res_species
            else:
                ratio = 1
            parray_scaled = parray * ratio
            darray_scaled = darray * ratio
            percent_scaled[pname]  = np.where(threshold_mask, parray_scaled, 0)
            density_scaled[pname] = np.where(threshold_mask, darray_scaled, 0)
    return device, baseline, stressor, threshold_exceeded, percent_scaled, density_scaled, rx, ry

def calculate_acoustic_stressors(
    fpath_dev,
    probabilities_file,
    paracousti_threshold_value,
    paracousti_metric,
    paracousti_weighting,
    fpath_nodev=None,
    species_folder=None,
    species_grid_resolution=None,
    latlon=True,
    Averaging=None,
):

    """
    Calculates the stressor layers from model and parameter input. Returns filepath to created rasters for dispaly in QGIS.

    Parameters
    ----------
    fpath_dev : str
        Directory path to the with device model run netcdf files.
    probabilities_file : str
        File path to probabilities/bondary condition *.csv file.
    paracousti_threshold_value : float
        threshold value (dB)
    paracousti_metric : str ['SEL'|'SPL']
        type of metric, either sound pressure level (SPL) or sound exposure level (SEL)
    paracousti_weighting : str
        species weighting used for calculation, must match variable name in .nc files.
    fpath_nodev : str, optional
        Directory path to the baseline/no device model run netcdf files. The default is None.
    species_folder : str, optional
        Directory path to the species files in the probabilities_file. The default is None.
    species_grid_resolution : float, defaults to None
        grid resolution of species_folder files
    latlon : Bool, optional
        True is coordinates are lat/lon. The default is True.
    Averaging : str
        type of depth selection to use. Options are 
            Depth Maximum : Maxmimum value along depth for each x,y (Default)
            Depth Average : Average value along depth for each x,y
            Bottom Bin : Value from bottom bin for each x,y
            Top Bin : Value from top (surface) bin for each x,y
            

    Returns
    -------
    dict_of_probabilistic_arrays : list
        2D arrays of:
            paracousti_without_devices: baseline_without_device,
            paracousti_with_devices: paracousti_with_device,
            paracousti_stressor: stressor,
            species_threshold_exceeded: threshold_exceeded,
            species_percent": percent_scaled,
            species_density": density_scaled,
    rx : array
        X-Coordiantes.
    ry : array
        Y-Coordinates.
    dx : scalar
        x-spacing.
    dy : scalar
        y-spacing.

    """
    # Ensure required files exist
    if not os.path.exists(fpath_dev):
        raise FileNotFoundError(f"The directory {fpath_dev} does not exist.")
    if not os.path.exists(probabilities_file):
        raise FileNotFoundError(f"The file {probabilities_file} does not exist.")

    paracousti_files = [
        os.path.join(fpath_dev, i) for i in os.listdir(fpath_dev) if i.endswith(".nc")
    ]
    conditions_probability = (
        pd.read_csv(probabilities_file).set_index("Paracousti File").fillna(0)
    )
    conditions_probability["% of yr"] = 100 * (
        conditions_probability["% of yr"] / conditions_probability["% of yr"].sum()
    )

    threshold = float(paracousti_threshold_value)
    if not (
        (species_grid_resolution is None)
        or (species_grid_resolution == "")
    ):
        grid_res_species = float(species_grid_resolution) * 1.0e6 # converted to m2
    else:
        grid_res_species = 0.0

    for ic, paracousti_file in enumerate(paracousti_files):
        with Dataset(paracousti_file) as ds:
            acoust_var = ds.variables[f"{paracousti_weighting}_{paracousti_metric}"][:].data
            cords = ds.variables[f"{paracousti_weighting}_{paracousti_metric}"].coordinates.split()
            X = ds.variables[cords[0]][:].data
            Y = ds.variables[cords[1]][:].data
            if X.shape[0] != acoust_var.shape[0]:
                acoust_var = np.transpose(acoust_var, (1, 2, 0))
            if ic == 0:
                xunits = ds.variables[cords[0]].units
                if "degrees" in xunits:
                    latlon = True
                    XCOR = np.where(X < 0, X + 360, X)
                else:
                    XCOR = X
                YCOR = Y
                Paracousti = np.zeros(
                    (
                        len(paracousti_files),
                        np.shape(acoust_var)[0],
                        np.shape(acoust_var)[1],
                        np.shape(acoust_var)[2],
                    )
                )
            Paracousti[ic, :] = acoust_var

    if not (
        (fpath_nodev is None) or (fpath_nodev == "")
    ):  # Assumes same grid as paracousti_files
        if not os.path.exists(fpath_nodev):
            raise FileNotFoundError(f"The directory {fpath_nodev} does not exist.")
        baseline_files = [
            os.path.join(fpath_nodev, i)
            for i in os.listdir(fpath_nodev)
            if i.endswith(".nc")
        ]
        for ic, baseline_file in enumerate(baseline_files):
            with Dataset(baseline_file) as ds:
                baseline = ds.variables[f"{paracousti_weighting}_{paracousti_metric}"][:].data
                cords = ds.variables[f"{paracousti_weighting}_{paracousti_metric}"].coordinates.split()
                if ds.variables[cords[0]][:].data.shape[0] != baseline.shape[0]:
                    baseline = np.transpose(baseline, (1, 2, 0))
                if ic == 0:
                    Baseline = np.zeros(
                        (
                            len(baseline_files),
                            np.shape(baseline)[0],
                            np.shape(baseline)[1],
                            np.shape(baseline)[2],
                        )
                    )
                Baseline[ic, :] = baseline
    else:
        Baseline = np.zeros(Paracousti.shape)

    if Averaging == "Depth Maximum":
        Paracousti = np.nanmax(Paracousti, axis=3)
        Baseline = np.nanmax(Baseline, axis=3)
    elif Averaging == "Depth Average":
        Paracousti = np.nanmean(Paracousti, axis=3)
        Baseline = np.nanmean(Baseline, axis=3)
    elif Averaging == "Bottom Bin":
        Paracousti = Paracousti[:, :, -1]
        Baseline = Baseline[:, :, -1]
    elif Averaging == "Top Bin":
        Paracousti = Paracousti[:, :, 0]
        Baseline = Baseline[:, :, 0]
    else:
        Paracousti = np.nanmax(Paracousti, axis=3)
        Baseline = np.nanmax(Baseline, axis=3)

    metric_calc = 'SPL' if 'spl'.casefold() in paracousti_metric.casefold() else 'SEL'
    
    paracousti_with_device, baseline_without_device, stressor, threshold_exceeded, percent_scaled, density_scaled, rx, ry = calc_probabilistic_metrics(
    paracousti_files,
    conditions_probability,
    threshold,
    Paracousti,
    Baseline,
    XCOR,
    YCOR,
    latlon,
    metric_calc = metric_calc,
    species_folder=species_folder,
    grid_res_species=grid_res_species
    )

    device_single, baseline_single, stressor_single, threshold_exceeded_single, percent_scaled_single, density_scaled_single, _, _ = calc_nonprobabilistic_metrics(
        paracousti_files,
        conditions_probability,
        threshold,
        Paracousti,
        Baseline,
        XCOR,
        YCOR,
        latlon,
        metric_calc = metric_calc,
        species_folder=species_folder,
        grid_res_species=grid_res_species,
        sel_hours=24,
    )

    dict_of_probabilistic_arrays = {
        "paracousti_without_devices": baseline_without_device,
        "paracousti_with_devices": paracousti_with_device,
        "paracousti_stressor": stressor,
        "species_threshold_exceeded": threshold_exceeded,
        "species_percent": percent_scaled,
        "species_density": density_scaled,
    }

    dict_of_nonprobabilistic_arrays = {
        "paracousti_without_devices": baseline_single,
        "paracousti_with_devices": device_single,
        "paracousti_stressor": stressor_single,
        "species_threshold_exceeded": threshold_exceeded_single,
        "species_percent": percent_scaled_single,
        "species_density": density_scaled_single,
    }
    

    dx = np.nanmean(np.diff(rx[0, :]))
    dy = np.nanmean(np.diff(ry[:, 0]))
    return dict_of_probabilistic_arrays, dict_of_nonprobabilistic_arrays, rx, ry, dx, dy

def create_output_rasters_nonprobabilistic(use_nonprobabilistic_arrays, dict_of_nonprobabilistic_arrays, crs, dx, dy, rx, ry, output_path):
    """Creates rasters for the defined non-probabilistic acoustic metrics.

    use_nonprobabilistic_arrays : list
        numpy arrays to generate rasters from
    dict_of_nonprobabilistic_arrays : dict
        list of all numpy arrays
    crs : int
        Coordiante Reference System / EPSG code.
    dx : scalar
        x-spacing.
    dy : scalar
        y-spacing.
    rx : array
        X-Coordiantes.
    ry : array
        Y-Coordinates.
    output_path : str | filepath
        file path to save individual rasters as GeoTIFF files
    Returns
    -------        
    output_rasters : dict
        output raster filenames with key indicating the calculation type and subkey indicating the associated paracousti filename
    """
    output_rasters = {}
    cell_resolution = [dx, dy]
    if crs == 4326:
        rxx = np.where(rx > 180, rx - 360, rx)
        bounds = [rxx.min() - dx / 2, ry.max() - dy / 2]
    else:
        bounds = [rx.min() - dx / 2, ry.max() - dy / 2]    
        
    for var in use_nonprobabilistic_arrays:
        output_rasters[var] = []
        for key in dict_of_nonprobabilistic_arrays[var].keys():
            array_name = var + "_" + key + ".tif" #file name of raster using analysis type and probability filename
            numpy_array = np.flip(dict_of_nonprobabilistic_arrays[var][key], axis=0)
            rows, cols = numpy_array.shape
            output_rasters[var].append(os.path.join(output_path, array_name))
            output_raster = create_raster(
                os.path.join(output_path, array_name), cols, rows, nbands=1
            )
            numpy_array_to_raster(
                output_raster,
                numpy_array,
                bounds,
                cell_resolution,
                crs,
                os.path.join(output_path, array_name),
            )
            output_raster = None
    return output_rasters

def create_output_rasters_probabilistic(use_probabilistic_arrays, dict_of_probabilistic_arrays, crs, dx, dy, rx, ry, output_path):
    """Creates rasters for the defined probabilistic acoustic metrics.

    use_probabilistic_arrays : list
        numpy arrays to generate rasters from
    dict_of_probabilistic_arrays : dict
        list of all numpy arrays
    crs : int
        Coordiante Reference System / EPSG code.
    dx : scalar
        x-spacing.
    dy : scalar
        y-spacing.
    rx : array
        X-Coordiantes.
    ry : array
        Y-Coordinates.
    output_path : str | filepath
        file path to save individual rasters as GeoTIFF files
    Returns
    -------        
    output_rasters : dict
        output raster filenames with key indicating the calculation type
    """
    numpy_array_names = [i + ".tif" for i in use_probabilistic_arrays]
    output_rasters = []
    cell_resolution = [dx, dy]
    if crs == 4326:
        rxx = np.where(rx > 180, rx - 360, rx)
        bounds = [rxx.min() - dx / 2, ry.max() - dy / 2]
    else:
        bounds = [rx.min() - dx / 2, ry.max() - dy / 2]    
        
    for array_name, use_numpy_array in zip(numpy_array_names, use_probabilistic_arrays):
        numpy_array = np.flip(dict_of_probabilistic_arrays[use_numpy_array], axis=0)
        rows, cols = numpy_array.shape
        # create an ouput raster given the stressor file path
        output_rasters.append(os.path.join(output_path, array_name))
        output_raster = create_raster(
            os.path.join(output_path, array_name), cols, rows, nbands=1
        )

        # post processing of numpy array to output raster
        numpy_array_to_raster(
            output_raster,
            numpy_array,
            bounds,
            cell_resolution,
            crs,
            os.path.join(output_path, array_name),
        )
        output_raster = None
    return output_rasters

def create_probabilistic_binned_csv(output_path, crs, secondary_constraint_filename=None, species_folder=None):
    """Calculates the binned statistics for the probabilistic calculations

    output_path : str | filepath
        file path to save individual csv files, also the location of the geotiff files
    crs : int
        Coordiante Reference System / EPSG code.
    secondary_constraint_filename: str | filename, defaults to None
        file name of the risk layer used.
    species_folder : str | filepath, defaults to None
        directory of species density and likelihood files.
    """
    #TODO : make secondary constraint and species folder bool

    vars = ["paracousti_without_devices", "paracousti_with_devices", "paracousti_stressor", "species_threshold_exceeded"]
    for var in vars:
        bin_layer(
            os.path.join(output_path, var + ".tif"), latlon=crs == 4326
            ).to_csv(os.path.join(output_path, var + ".csv"), index=False)

    if not (
        (secondary_constraint_filename is None) or (secondary_constraint_filename == "")
    ):
        bin_layer(
            os.path.join(output_path, "paracousti_stressor.tif"),
            receptor_filename=os.path.join(output_path, "paracousti_risk_layer.tif"),
            receptor_names=None,
            limit_receptor_range=[0, np.inf],
            latlon=crs == 4326,
        ).to_csv(
            os.path.join(
                output_path, "paracousti_stressor_at_paracousti_risk_layer.csv"
            ),
            index=False,
        )

    if not ((species_folder is None) or (species_folder == "")):
        vars = ["species_percent", "species_density"]
        for var in vars:
            bin_layer(
                os.path.join(output_path, var + ".tif"), latlon=crs == 4326
                ).to_csv(os.path.join(output_path, var + ".csv"), index=False)

        if not (
            (secondary_constraint_filename is None)
            or (secondary_constraint_filename == "")
        ):

            vars = ["species_threshold_exceeded", "species_percent", "species_density"]
            for var in vars:
                bin_layer(
                    os.path.join(output_path, var + ".tif"),
                    receptor_filename=os.path.join(output_path, "paracousti_risk_layer.tif"),
                    receptor_names=None,
                    limit_receptor_range=[0, np.inf],
                    latlon=crs == 4326,
                ).to_csv(
                    os.path.join(
                        output_path, var + "_at_paracousti_risk_layer.csv"
                    ),
                    index=False,
                )

def create_nonprobabilistic_binned_csv(output_path, output_rasters, crs, secondary_constraint_filename=None, species_folder=None):
    """Calculates the binned statistics for the non-probabilistic calculations

    output_path : str | filepath
        file path to save individual csv files, also the location of the geotiff files
    output_rasters: str | filename
        filenames of the non-probablistic GeoTIFs
    crs : int
        Coordiante Reference System / EPSG code.
    secondary_constraint_filename: str | filename, defaults to None
        file name of the risk layer used.
    species_folder : str | filepath, defaults to None
        directory of species density and likelihood files.
    """
    #TODO : make secondary constraint and species folder bool
    vars = ["paracousti_without_devices", "paracousti_with_devices", "paracousti_stressor", "species_threshold_exceeded"]
    for var in vars:
        for file in output_rasters[var]:
            bin_layer(
                os.path.join(output_path, file), latlon=crs == 4326
                ).to_csv(os.path.join(output_path, file.split('.tif')[0] + ".csv"), index=False)
            
    if not ((species_folder is None) or (species_folder == "")):
        vars = ["species_percent", "species_density", "paracousti_stressor", "species_threshold_exceeded"]
        for var in vars:
            for file in output_rasters[var]:
                bin_layer(
                    os.path.join(output_path, file), latlon=crs == 4326
                    ).to_csv(os.path.join(output_path,file.split('.tif')[0] + ".csv"), index=False)

    if not (
        (secondary_constraint_filename is None) or (secondary_constraint_filename == "")
    ):
        vars = ["paracousti_stressor", "species_threshold_exceeded", "species_percent", "species_density"]
        for var in vars:
            for file in output_rasters[var]:
                bin_layer(
                    os.path.join(output_path, file),
                    receptor_filename=os.path.join(output_path, "paracousti_risk_layer.tif"),
                    receptor_names=None,
                    limit_receptor_range=[0, np.inf],
                    latlon=crs == 4326,
                ).to_csv(
                    os.path.join(
                        output_path, file.split('.tif')[0] + "_at_paracousti_risk_layer.csv"
                    ),
                    index=False,
                )

def run_acoustics_stressor(
    dev_present_file,
    dev_notpresent_file,
    probabilities_file,
    crs,
    output_path,
    paracousti_threshold_value,
    paracousti_weighting,
    paracousti_metric,
    species_folder=None,
    paracousti_species_grid_resolution=None,
    Averaging=None,
    secondary_constraint_filename=None,
):
    """_summary_

    Parameters
    ----------
    dev_present_file : str
        Directory path to the baseline/no device model run netcdf files.
    dev_notpresent_file : str
        Directory path to the baseline/no device model run netcdf files.
    probabilities_file : str
        File path to probabilities/bondary condition *.csv file..
    crs : int
        Coordiante Reference System / EPSG code.
    output_path : str
        File directory to save output.
    paracousti_threshold_value : float
        threshold value (dB)
    paracousti_weighting : str
        species weighting used for calculation, must match variable name in .nc files.
    paracousti_metric : str ['SEL'|'SPL']
        type of metric, either sound pressure level (SPL) or sound exposure level (SEL)
    species_folder : str, optional
        Directory path to the species files in the probabilities_file. The default is None.
    paracousti_species_grid_resolution : float, defaults to None
        grid resolution of species_folder files
    Averaging : str
        type of depth selection to use. Options are 
            Depth Maximum : Maxmimum value along depth for each x,y (Default)
            Depth Average : Average value along depth for each x,y
            Bottom Bin : Value from bottom bin for each x,y
            Top Bin : Value from top (surface) bin for each x,y
    secondary_constraint_filename: str | filename, defaults to None
        file name of the risk layer used.

    
    Raises 
    ------
    FileNotFoundError: secondary_constraint_filename entered does not exist

    Returns
    -------
    OUTPUT_probabilistic : dict
        rasters filenames for the probabilistic calculations
        keys are names of probabilistic output rasters without 
        extension and values are the filename:
        keys include :
            paracousti_without_devices
            paracousti_with_devices
            paracousti_stressor
        if species_folder present:
            species_threshold_exceeded
            species_percent
            species_density
        if secondary_constraint_filename
            risk_layer
            
    OUTOUT_nonprobabilistic : dict
        raster filenames for the non-probabilistic calculations
        keys are names of probabilistic output rasters without 
        extension and values are the filename. Subkeys are the assocatged
        paracousti files without extension.
        keys include :
            paracousti_without_devices
            paracousti_with_devices
            paracousti_stressor
        if species_folder present:
            species_threshold_exceeded
            species_percent
            species_density
"""

    os.makedirs(
        output_path, exist_ok=True
    )  # create output directory if it doesn't exist

    dict_of_probabilistic_arrays, dict_of_nonprobabilistic_arrays, rx, ry, dx, dy = calculate_acoustic_stressors(
        fpath_dev=dev_present_file,
        probabilities_file=probabilities_file,
        paracousti_threshold_value=paracousti_threshold_value,
        paracousti_metric=paracousti_metric,
        paracousti_weighting=paracousti_weighting,
        fpath_nodev=dev_notpresent_file,
        species_folder=species_folder,
        species_grid_resolution=paracousti_species_grid_resolution,
        latlon=crs == 4326,
        Averaging=Averaging,
    )

    if not ((species_folder is None) or (species_folder == "")):
        use_probabilistic_arrays = [
            "paracousti_without_devices",
            "paracousti_with_devices",
            "paracousti_stressor",
            "species_threshold_exceeded",
            "species_percent",
            "species_density",
        ]
        use_nonprobabilistic_arrays = [
            "paracousti_without_devices",
            "paracousti_with_devices",
            "paracousti_stressor",
            "species_threshold_exceeded",
            "species_percent",
            "species_density",
        ]
    else:
        use_probabilistic_arrays = [
            "paracousti_without_devices",
            "paracousti_with_devices",
            "paracousti_stressor",
            "species_threshold_exceeded",
        ]
        use_nonprobabilistic_arrays = [
            "paracousti_without_devices",
            "paracousti_with_devices",
            "paracousti_stressor",
            "species_threshold_exceeded",
        ]
    if not (
        (secondary_constraint_filename is None) or (secondary_constraint_filename == "")
    ):
        if not os.path.exists(secondary_constraint_filename):
            raise FileNotFoundError(
                f"The file {secondary_constraint_filename} does not exist."
            )
        rrx, rry, constraint = secondary_constraint_geotiff_to_numpy(
            secondary_constraint_filename
        )
        dict_of_probabilistic_arrays["paracousti_risk_layer"] = resample_structured_grid(
            rrx, rry, constraint, rx, ry, interpmethod="nearest"
        )
        use_probabilistic_arrays.append("paracousti_risk_layer")

    output_rasters_probabilistic = create_output_rasters_probabilistic(use_probabilistic_arrays, dict_of_probabilistic_arrays, crs, dx, dy, rx, ry, output_path)
    create_probabilistic_binned_csv(output_path, crs, secondary_constraint_filename=secondary_constraint_filename, species_folder=species_folder)
    
    output_rasters_nonprobabilistic = create_output_rasters_nonprobabilistic(use_nonprobabilistic_arrays, dict_of_nonprobabilistic_arrays, crs, dx, dy, rx, ry, output_path)
    create_nonprobabilistic_binned_csv(output_path, output_rasters_nonprobabilistic, crs, secondary_constraint_filename=secondary_constraint_filename, species_folder=species_folder)


    OUTPUT_probabilistic = {}
    for val in output_rasters_probabilistic:
        OUTPUT_probabilistic[os.path.basename(os.path.normpath(val)).split(".")[0]] = val

    OUTOUT_nonprobabilistic = {}
    for var_fname in output_rasters_nonprobabilistic.keys():
        var = os.path.basename(os.path.normpath(var_fname)).split(".")[0]
        OUTOUT_nonprobabilistic[var] = {}
        for val in output_rasters_nonprobabilistic[var]:
            OUTOUT_nonprobabilistic[var][val] = val
    return OUTPUT_probabilistic, OUTOUT_nonprobabilistic

