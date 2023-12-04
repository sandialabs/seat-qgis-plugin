"""
/***************************************************************************.

 shear_stress_module.py
 Copyright 2023, Integral Consulting Inc. All rights reserved.
 
 PURPOSE: module for calcualting shear stress (sediment mobility) change from a shear stress stressor

 PROJECT INFORMATION:
 Name: SEAT - Spatial and Environmental Assessment Toolkit
 Number: C1308

 AUTHORS
  Timothy Nelson (tnelson@integral-corp.com)
  Sam McWilliams (smcwilliams@integral-corp.com)
  Eben Pendelton
 
 NOTES (Data descriptions and any script specific notes)
	1. called by stressor_receptor_calc.py
"""

import os

import numpy as np
import pandas as pd
from netCDF4 import Dataset

from .stressor_utils import (
    estimate_grid_spacing,
    create_structured_array_from_unstructured,
    calc_receptor_array,
    trim_zeros,
    create_raster,
    numpy_array_to_raster,
    classify_layer_area,
    bin_layer
)


def critical_shear_stress(D_meters, rhow=1024, nu=1e-6, s=2.65, g=9.81):
    """
    Calculate critical shear stress from grain size.

    Parameters
    ----------
    D_meters : Array
        grain size in meters.
    rhow : scalar, optional
        density of water in kg/m3. The default is 1024.
    nu : scalar, optional
        kinematic viscosity of water. The default is 1e-6.
    s : scalar, optional
        specific gravity of sediment. The default is 2.65.
    g : scalar, optional
        acceleratin due to gravity. The default is 9.81.

    Returns
    -------
    taucrit : TYPE
        DESCRIPTION.

    """
    Dstar = ((g * (s-1)) / nu**2)**(1/3) * D_meters
    SHcr = (0.3/(1+1.2*Dstar)) + 0.055*(1-np.exp(-0.02 * Dstar))
    taucrit = rhow * (s - 1) * g * D_meters * SHcr  # in Pascals
    return taucrit


def classify_mobility(mobility_parameter_dev, mobility_parameter_nodev):
    """
    classifies sediment mobility from device runs to no device runs.

    Parameters
    ----------
    mobility_parameter_dev : Array
        mobility parameter (tau/tau_crit) for with device runs.
    mobility_parameter_nodev : TYPE
        mobility parameter (tau/tau_crit) for without (baseline) device runs.

    Returns
    -------
    mobility_classification : array
        Numerically classified array where,
        3 = New Erosion
        2 = Increased Erosion
        1 = Reduced Erosion
        0 = No Change
        -1 = Reduced Deposition
        -2 = Increased Deposition
        -3 = New Deposition
    """

    mobility_classification = np.zeros(mobility_parameter_dev.shape)
    # New Erosion = 3
    mobility_classification = np.where(((mobility_parameter_nodev < mobility_parameter_dev) & (
        mobility_parameter_nodev < 1) & (mobility_parameter_dev >= 1)), 3, mobility_classification)
    # Increased Erosion (Tw>Tb) & (Tw-Tb)>1 = 2
    mobility_classification = np.where(((mobility_parameter_dev > mobility_parameter_nodev) & (
        mobility_parameter_nodev >= 1) & (mobility_parameter_dev >= 1)), 2, mobility_classification)
    # Reduced Erosion (Tw<Tb) & (Tw-Tb)>1 = 1
    mobility_classification = np.where(((mobility_parameter_dev < mobility_parameter_nodev) & (
        mobility_parameter_nodev >= 1) & (mobility_parameter_dev >= 1)), 1, mobility_classification)
    # Reduced Deposition (Tw>Tb) & (Tw-Tb)<1 = -1
    mobility_classification = np.where(((mobility_parameter_dev > mobility_parameter_nodev) & (
        mobility_parameter_nodev < 1) & (mobility_parameter_dev < 1)), -1, mobility_classification)
    # Increased Deposition (Tw>Tb) & (Tw-Tb)>1 = -2
    mobility_classification = np.where(((mobility_parameter_dev < mobility_parameter_nodev) & (
        mobility_parameter_nodev < 1) & (mobility_parameter_dev < 1)), -2, mobility_classification)
    # New Deposition = -3
    mobility_classification = np.where(((mobility_parameter_dev < mobility_parameter_nodev) & (
        mobility_parameter_nodev >= 1) & (mobility_parameter_dev < 1)), -3, mobility_classification)
    # NoChange = 0
    return mobility_classification


def check_grid_define_vars(dataset):
    """
    Determins the type of grid and corresponding shear stress variable name and coordiante names

    Parameters
    ----------
    dataset : netdcf (.nc) dataset
        netdcf (.nc) dataset.

    Returns
    -------
    gridtype : string
        "structured" or "unstructured".
    xvar : str
        name of x-coordinate variable.
    yvar : str
        name of y-coordiante variable.
    tauvar : str
        name of shear stress variable.

    """
    vars = list(dataset.variables)
    if 'TAUMAX' in vars:
        gridtype = 'structured'
        tauvar = 'TAUMAX'
    else:
        gridtype = 'unstructured'
        tauvar = 'taus'
    xvar, yvar = dataset.variables[tauvar].coordinates.split()
    return gridtype, xvar, yvar, tauvar


def calculate_shear_stress_stressors(fpath_nodev,
                                     fpath_dev,
                                     probabilities_file,
                                     receptor_filename=None,
                                     latlon=True,
                                     value_selection='MAX'
                                     ):
    """
    Calculates the stressor layers as arrays from model and parameter input.

    Parameters
    ----------
    fpath_nodev : str
        Directory path to the baseline/no device model run netcdf files.
    fpath_dev : str
        Directory path to the with device model run netcdf files.
    probabilities_file : str
        File path to probabilities/bondary condition *.csv file.
    receptor_filename : str, optional
        File path to the recetptor file (*.csv or *.tif). The default is None.
    latlon : Bool, optional
        True is coordinates are lat/lon. The default is True.
    value_selection : str, optional
        Temporal selection of shears stress (not currently used). The default is 'MAX'.

    Raises
    ------
    Exception
        "Number of device runs files must be the same as no device runs files".

    Returns
    -------
    listOfFiles : list
        2D arrays of:
        [0] tau_diff
        [1] mobility_parameter_nodev
        [2] mobility_parameter_dev
        [3] mobility_parameter_diff
        [4] mobility_classification
        [5] receptor array  
        [6] tau_combined_dev
        [7] tau_combined_nodev
    rx : array
        X-Coordiantes.
    ry : array
        Y-Coordinates.
    dx : scalar
        x-spacing.
    dy : scalar
        y-spacing.
    gridtype : str
        grid type [structured or unstructured].

    """

    files_nodev = [i for i in os.listdir(fpath_nodev) if i.endswith('.nc')]
    files_dev = [i for i in os.listdir(fpath_dev) if i.endswith('.nc')]

    # Load and sort files
    if len(files_nodev) == 1 & len(files_dev) == 1:
        # asumes a concatonated files with shape
        # [run_num, time, rows, cols]

        file_dev_present = Dataset(os.path.join(fpath_dev, files_dev[0]))
        gridtype, xvar, yvar, tauvar = check_grid_define_vars(file_dev_present)
        xcor = file_dev_present.variables[xvar][:].data
        ycor = file_dev_present.variables[yvar][:].data
        tau_dev = file_dev_present.variables[tauvar][:]
        # close the device prsent file
        file_dev_present.close()

        file_dev_notpresent = Dataset(
            os.path.join(fpath_nodev, files_nodev[0]))
        tau_nodev = file_dev_notpresent.variables[tauvar][:]
        # close the device not present file
        file_dev_notpresent.close()

        # if tau_dev.shape[0] != tau_nodev.shape[0]:
        #     raise Exception(f"Number of device runs ({tau_dev.shape[0]}) must be the same as no device runs ({tau_nodev.shape[0]}).")

    # same number of files, file name must be formatted with either run number
    elif len(files_nodev) == len(files_dev):
        # asumes each run is separate with the some_name_RunNum_map.nc, where run number comes at the last underscore before _map.nc
        run_num_nodev = np.zeros((len(files_nodev)))
        for ic, file in enumerate(files_nodev):
            run_num_nodev[ic] = int(file.split('.')[0].split('_')[-2])
        run_num_dev = np.zeros((len(files_dev)))
        for ic, file in enumerate(files_dev):
            run_num_dev[ic] = int(file.split('.')[0].split('_')[-2])

        # ensure run oder for nodev matches dev files
        if np.any(run_num_nodev != run_num_dev):
            adjust_dev_order = []
            for ri in run_num_nodev:
                adjust_dev_order = np.append(
                    adjust_dev_order, np.flatnonzero(run_num_dev == ri))
            files_dev = [files_dev[int(i)] for i in adjust_dev_order]
            run_num_dev = [run_num_dev[int(i)] for i in adjust_dev_order]
        DF = pd.DataFrame({'files_nodev': files_nodev,
                           'run_num_nodev': run_num_nodev,
                           'files_dev': files_dev,
                           'run_num_dev': run_num_dev})
        DF = DF.sort_values(by='run_num_dev')

        first_run = True
        ir = 0
        for _, row in DF.iterrows():
            file_dev_notpresent = Dataset(
                os.path.join(fpath_nodev, row.files_nodev))
            file_dev_present = Dataset(os.path.join(fpath_dev, row.files_dev))

            gridtype, xvar, yvar, tauvar = check_grid_define_vars(
                file_dev_present)

            if first_run:
                tmp = file_dev_notpresent.variables[tauvar][:].data
                if gridtype == 'structured':
                    tau_nodev = np.zeros(
                        (DF.shape[0], tmp.shape[0], tmp.shape[1], tmp.shape[2]))
                    tau_dev = np.zeros(
                        (DF.shape[0], tmp.shape[0], tmp.shape[1], tmp.shape[2]))
                else:
                    tau_nodev = np.zeros(
                        (DF.shape[0], tmp.shape[0], tmp.shape[1]))
                    tau_dev = np.zeros(
                        (DF.shape[0], tmp.shape[0], tmp.shape[1]))
                xcor = file_dev_notpresent.variables[xvar][:].data
                ycor = file_dev_notpresent.variables[yvar][:].data
                first_run = False
            tau_nodev[ir, :] = file_dev_notpresent.variables[tauvar][:].data
            tau_dev[ir, :] = file_dev_present.variables[tauvar][:].data

            file_dev_notpresent.close()
            file_dev_present.close()
            ir += 1
    else:
        raise Exception(
            f"Number of device runs ({len(files_dev)}) must be the same as no device runs ({len(files_nodev)}).")
    # Finished loading and sorting files

    if (gridtype == 'structured'):
        if (xcor[0, 0] == 0) & (xcor[-1, 0] == 0):
            # at least for some runs the boundary has 0 coordinates. Check and fix.
            xcor, ycor, tau_nodev, tau_dev = trim_zeros(
                xcor, ycor, tau_nodev, tau_dev)

    if not (probabilities_file == ""):
        # Load BC file with probabilities and find appropriate probability
        BC_probability = pd.read_csv(probabilities_file, delimiter=",")
        BC_probability['run_num'] = BC_probability['run number']-1
        BC_probability = BC_probability.sort_values(by='run number')
        BC_probability["probability"] = BC_probability["% of yr"].values/100
        # BC_probability
        if 'Exclude' in BC_probability.columns:
            BC_probability = BC_probability[~(
                (BC_probability['Exclude'] == 'x') | (BC_probability['Exclude'] == 'X'))]
    else:  # assume run_num in file name is return interval
        BC_probability = pd.DataFrame()
        # ignore number and start sequentially from zero
        BC_probability['run_num'] = np.arange(0, tau_dev.shape[0])
        # assumes run_num in name is the return interval
        BC_probability["probability"] = 1/DF.run_num_dev.to_numpy()
        BC_probability["probability"] = BC_probability["probability"] / \
            BC_probability["probability"].sum()  # rescale to ensure = 1 

    # Calculate Stressor and Receptors
    # data_dev_max = np.amax(data_dev, axis=1, keepdims=True) #look at maximum shear stress difference change
    if value_selection == 'MAX':
        tau_dev = np.nanmax(tau_dev, axis=1, keepdims=True)  # max over time
        tau_nodev = np.nanmax(
            tau_nodev, axis=1, keepdims=True)  # max over time
    elif value_selection == 'MEAN':
        tau_dev = np.nanmean(tau_dev, axis=1, keepdims=True)  # mean over time
        tau_nodev = np.nanmean(
            tau_nodev, axis=1, keepdims=True)  # mean over time
    elif value_selection == 'LAST':
        tau_dev = tau_dev[:, -2:-1, :]  # bottom bin over time
        tau_nodev = tau_nodev[:, -2:-1, :]  # bottom bin over time
    else:
        tau_dev = np.nanmax(tau_dev, axis=1, keepdims=True)  # max over time
        tau_nodev = np.nanmax(
            tau_nodev, axis=1, keepdims=True)  # max over time

    # initialize arrays
    if gridtype == 'structured':
        tau_combined_nodev = np.zeros(np.shape(tau_nodev[0, 0, :, :]))
        tau_combined_dev = np.zeros(np.shape(tau_dev[0, 0, :, :]))
    else:
        tau_combined_nodev = np.zeros(np.shape(tau_nodev)[-1])
        tau_combined_dev = np.zeros(np.shape(tau_dev)[-1])

    for run_number, prob in zip(BC_probability['run_num'].values,
                                BC_probability["probability"].values):

        tau_combined_nodev = tau_combined_nodev + \
            prob * tau_nodev[run_number, -1, :]
        tau_combined_dev = tau_combined_dev + prob * tau_dev[run_number, -1, :]

    receptor_array = calc_receptor_array(
        receptor_filename, xcor, ycor, latlon=latlon)
    taucrit = critical_shear_stress(D_meters=receptor_array * 1e-6,
                                    rhow=1024,
                                    nu=1e-6,
                                    s=2.65,
                                    g=9.81)  # units N/m2 = Pa
    tau_diff = tau_combined_dev - tau_combined_nodev
    mobility_parameter_nodev = tau_combined_nodev / taucrit
    mobility_parameter_nodev = np.where(
        receptor_array == 0, 0, mobility_parameter_nodev)
    mobility_parameter_dev = tau_combined_dev / taucrit
    mobility_parameter_dev = np.where(
        receptor_array == 0, 0, mobility_parameter_dev)
    # Calculate risk metrics over all runs

    mobility_parameter_diff = mobility_parameter_dev - mobility_parameter_nodev

    if gridtype == 'structured':
        mobility_classification = classify_mobility(
            mobility_parameter_dev, mobility_parameter_nodev)
        dx = np.nanmean(np.diff(xcor[:, 0]))
        dy = np.nanmean(np.diff(ycor[0, :]))
        rx = xcor
        ry = ycor
        listOfFiles = [tau_diff, mobility_parameter_nodev, mobility_parameter_dev, mobility_parameter_diff,
                       mobility_classification, receptor_array, tau_combined_dev, tau_combined_nodev]
    else:  # unstructured
        dxdy = estimate_grid_spacing(xcor, ycor, nsamples=100)
        dx = dxdy
        dy = dxdy
        rx, ry, tau_diff_struct = create_structured_array_from_unstructured(
            xcor, ycor, tau_diff, dxdy, flatness=0.2)
        _, _, tau_combined_dev_struct = create_structured_array_from_unstructured(
            xcor, ycor, tau_combined_dev, dxdy, flatness=0.2)
        _, _, tau_combined_nodev_struct = create_structured_array_from_unstructured(
            xcor, ycor, tau_combined_nodev, dxdy, flatness=0.2)
        if not ((receptor_filename is None) or (receptor_filename == "")):
            _, _, mobility_parameter_nodev_struct = create_structured_array_from_unstructured(
                xcor, ycor, mobility_parameter_nodev, dxdy, flatness=0.2)
            _, _, mobility_parameter_dev_struct = create_structured_array_from_unstructured(
                xcor, ycor, mobility_parameter_dev, dxdy, flatness=0.2)
            _, _, mobility_parameter_diff_struct = create_structured_array_from_unstructured(
                xcor, ycor, mobility_parameter_diff, dxdy, flatness=0.2)
            _, _, receptor_array_struct = create_structured_array_from_unstructured(
                xcor, ycor, receptor_array, dxdy, flatness=0.2)
        else:
            mobility_parameter_nodev_struct = np.nan * tau_diff_struct
            mobility_parameter_dev_struct = np.nan * tau_diff_struct
            mobility_parameter_diff_struct = np.nan * tau_diff_struct
            receptor_array_struct = np.nan * tau_diff_struct
        mobility_classification = classify_mobility(
            mobility_parameter_dev_struct, mobility_parameter_nodev_struct)
        mobility_classification = np.where(
            np.isnan(tau_diff_struct), -100, mobility_classification)
        listOfFiles = [tau_diff_struct, mobility_parameter_nodev_struct, mobility_parameter_dev_struct, mobility_parameter_diff_struct,
                       mobility_classification, receptor_array_struct, tau_combined_dev_struct, tau_combined_nodev_struct]

    return listOfFiles, rx, ry, dx, dy, gridtype


def run_shear_stress_stressor(
    dev_present_file,
    dev_notpresent_file,
    probabilities_file,
    crs,
    output_path,
    receptor_filename=None
):
    """
    creates geotiffs and area change statistics files for shear stress change

    Parameters
    ----------
    dev_present_file : str
        Directory path to the baseline/no device model run netcdf files.
    dev_notpresent_file : str
        Directory path to the baseline/no device model run netcdf files.
    probabilities_file : str
        File path to probabilities/bondary condition *.csv file.
    crs : scalar
        Coordiante Reference System / EPSG code.
    output_path : str
        File directory to save output.
    receptor_filename : str, optional
        File path to the recetptor file (*.csv or *.tif). The default is None.

    Returns
    -------
    output_rasters : list
        names of output rasters:
        [0] 'calculated_stressor.tif'
        [1] or [6] 'tau_dev.tif'
        [2] or [7] 'tau_nodev.tif'
        if receptor present:
            [1] 'calculated_stressor_with_receptor.tif'
            [2] 'calculated_stressor_reclassified.tif'
            [3] 'receptor.tif'

    """

    numpy_arrays, rx, ry, dx, dy, gridtype = calculate_shear_stress_stressors(fpath_nodev=dev_notpresent_file,
                                                                              fpath_dev=dev_present_file,
                                                                              probabilities_file=probabilities_file,
                                                                              receptor_filename=receptor_filename,
                                                                              latlon=crs == 4326)
    # numpy_arrays = [0] tau_diff
    #               [1] mobility_parameter_nodev
    #               [2] mobility_parameter_dev
    #               [3] mobility_parameter_diff
    #               [4] mobility_classification
    #               [5] receptor array
    #               [6] tau_combined_dev
    #               [7] tau_combined_nodev
    if not ((receptor_filename is None) or (receptor_filename == "")):
        numpy_array_names = ['calculated_stressor.tif',
                             'calculated_stressor_with_receptor.tif',
                             'calculated_stressor_reclassified.tif',
                             'receptor.tif',
                             'tau_with_devices.tif',
                             'tau_without_devices.tif']
        use_numpy_arrays = [numpy_arrays[0], numpy_arrays[3],
                            numpy_arrays[4], numpy_arrays[5], numpy_arrays[6], numpy_arrays[7]]
    else:
        numpy_array_names = ['calculated_stressor.tif',
                             'tau_with_devices.tif',
                             'tau_without_devices.tif']
        use_numpy_arrays = [numpy_arrays[0], numpy_arrays[6], numpy_arrays[7]]

    output_rasters = []
    for array_name, numpy_array in zip(numpy_array_names, use_numpy_arrays):

        if gridtype == 'structured':
            numpy_array = np.flip(np.transpose(numpy_array), axis=0)
        else:
            numpy_array = np.flip(numpy_array, axis=0)

        cell_resolution = [dx, dy]
        if crs == 4326:
            rxx = np.where(rx > 180, rx-360, rx)
            bounds = [rxx.min() - dx/2, ry.max() - dy/2]
        else:
            bounds = [rx.min() - dx/2, ry.max() - dy/2]
        rows, cols = numpy_array.shape
        # create an ouput raster given the stressor file path
        output_rasters.append(os.path.join(output_path, array_name))
        output_raster = create_raster(
            os.path.join(output_path, array_name),
            cols,
            rows,
            nbands=1,
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

    # Area calculations pull form rasters to ensure uniformity
    bin_layer(os.path.join(output_path, numpy_array_names[0]),
              receptor_filename=None,
              receptor_names=None,
              latlon=crs == 4326).to_csv(os.path.join(output_path, "calculated_stressor.csv"), index=False)
    if not ((receptor_filename is None) or (receptor_filename == "")):
        bin_layer(os.path.join(output_path, numpy_array_names[0]),
                  receptor_filename=os.path.join(
                      output_path, numpy_array_names[3]),
                  receptor_names=None,
                  limit_receptor_range=[0, np.inf],
                  latlon=crs == 4326).to_csv(os.path.join(output_path, "calculated_stressor_at_receptor.csv"), index=False)
        bin_layer(os.path.join(output_path, numpy_array_names[1]),
                  receptor_filename=os.path.join(
                      output_path, numpy_array_names[3]),
                  receptor_names=None,
                  limit_receptor_range=[0, np.inf],
                  latlon=crs == 4326).to_csv(os.path.join(output_path, "calculated_stressor_with_receptor.csv"), index=False)

        classify_layer_area(os.path.join(output_path, "calculated_stressor_reclassified.tif"),
                            at_values=[-3, -2, -1, 0, 1, 2, 3],
                            value_names=['New Deposition', 'Increased Deposition', 'Reduced Deposition',
                                         'No Change', 'Reduced Erosion', 'Increased Erosion', 'New Erosion'],
                            latlon=crs == 4326).to_csv(os.path.join(output_path, "calculated_stressor_reclassified.csv"), index=False)

        classify_layer_area(os.path.join(output_path, "calculated_stressor_reclassified.tif"),
                            receptor_filename=os.path.join(
                                output_path, numpy_array_names[3]),
                            at_values=[-3, -2, -1, 0, 1, 2, 3],
                            value_names=['New Deposition', 'Increased Deposition', 'Reduced Deposition',
                                         'No Change', 'Reduced Erosion', 'Increased Erosion', 'New Erosion'],
                            limit_receptor_range=[0, np.inf],
                            latlon=crs == 4326).to_csv(os.path.join(output_path, "calculated_stressor_reclassified_at_receptor.csv"), index=False)

    return output_rasters
