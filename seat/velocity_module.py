#!/usr/bin/python
"""
/***************************************************************************.

 velocity_module.py
 Copyright 2023, Integral Consulting Inc. All rights reserved.
 
 PURPOSE: module for calcualting velocity (larval motility) change from a velocity stressor

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

import glob
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
    bin_layer,
    classify_layer_area,
    classify_layer_area_2nd_Constraint,
    resample_structured_grid,
    secondary_constraint_geotiff_to_numpy
)


def classify_motility(motility_parameter_dev, motility_parameter_nodev):
    """
    classifies larval motility from device runs to no device runs.

    Parameters
    ----------
    motility_parameter_dev : Array
        motility parameter (vel/vel_crit) for with device runs.
    motility_parameter_nodev : TYPE
        motility parameter (vel/vel_crit) for without (baseline) device runs.

    Returns
    -------
    motility_classification : array
        Numerically classified array where,
        3 = New Motility
        2 = Increased Motility
        1 = Reduced Motility
        0 = No Change
        -1 = Motility Stops
    """

    motility_classification = np.zeros(motility_parameter_dev.shape)
    # Motility Stops
    motility_classification = np.where(((motility_parameter_dev < motility_parameter_nodev) & (
        motility_parameter_nodev >= 1) & (motility_parameter_dev < 1)), -1, motility_classification)
    # Reduced Motility (Tw<Tb) & (Tw-Tb)>1
    motility_classification = np.where(((motility_parameter_dev < motility_parameter_nodev) & (
        motility_parameter_nodev >= 1) & (motility_parameter_dev >= 1)), 1, motility_classification)
    # Increased Motility (Tw>Tb) & (Tw-Tb)>1
    motility_classification = np.where(((motility_parameter_dev > motility_parameter_nodev) & (
        motility_parameter_nodev >= 1) & (motility_parameter_dev >= 1)), 2, motility_classification)
    # New Motility
    motility_classification = np.where(((motility_parameter_dev > motility_parameter_nodev) & (
        motility_parameter_nodev < 1) & (motility_parameter_dev >= 1)), 3, motility_classification)
    # NoChange or NoMotility = 0
    return motility_classification


def check_grid_define_vars(dataset):
    """
    Determins the type of grid and corresponding velocity variable name and coordiante names

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
    uvar : str
        name of x-coordinate velocity variable.
    vvar : str
        name of y-coordinate velocity variable.
    """
    vars = list(dataset.variables)
    if 'U1' in vars:
        gridtype = 'structured'
        uvar = 'U1'
        vvar = 'V1'
        try:
            xvar, yvar = dataset.variables[uvar].coordinates.split()
        except:
            xvar = 'XCOR'
            yvar = 'YCOR'
    else:
        gridtype = 'unstructured'
        uvar = 'ucxa'
        vvar = 'ucya'
        xvar, yvar = dataset.variables[uvar].coordinates.split()
    return gridtype, xvar, yvar, uvar, vvar


def calculate_velocity_stressors(fpath_nodev,
                                 fpath_dev,
                                 probabilities_file,
                                 receptor_filename=None,
                                 latlon=True,
                                 value_selection='MAX'
                                 ):
    """


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
            [0] mag_diff
            [1] motility_nodev
            [2] motility_dev
            [3] motility_diff
            [4] motility_classification    
            [5] receptor (vel_crit)
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
        gridtype, xvar, yvar, uvar, vvar = check_grid_define_vars(
            file_dev_present)
        xcor = file_dev_present.variables[xvar][:].data
        ycor = file_dev_present.variables[yvar][:].data
        u = file_dev_present.variables[uvar][:].data
        v = file_dev_present.variables[vvar][:].data
        mag_dev = np.sqrt(u**2 + v**2)

        # close the device prsent file
        file_dev_present.close()

        file_dev_notpresent = Dataset(
            os.path.join(fpath_nodev, files_nodev[0]))
        u = file_dev_notpresent.variables[uvar][:].data
        v = file_dev_notpresent.variables[vvar][:].data
        mag_nodev = np.sqrt(u**2 + v**2)
        # close the device not present file
        file_dev_notpresent.close()

        # if mag_dev.shape[0] != mag_nodev.shape[0]:
        #     raise Exception(f"Number of device runs ({mag_dev.shape[0]}) must be the same as no device runs ({mag_nodev.shape[0]}).")

    # same number of files, file name must be formatted with either run number or return interval
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

            gridtype, xvar, yvar, uvar, vvar = check_grid_define_vars(
                file_dev_present)

            if first_run:
                tmp = file_dev_notpresent.variables[uvar][:].data
                if gridtype == 'structured':
                    mag_nodev = np.zeros(
                        (DF.shape[0], tmp.shape[0], tmp.shape[1], tmp.shape[2], tmp.shape[3]))
                    mag_dev = np.zeros(
                        (DF.shape[0], tmp.shape[0], tmp.shape[1], tmp.shape[2], tmp.shape))
                else:
                    mag_nodev = np.zeros(
                        (DF.shape[0], tmp.shape[0], tmp.shape[1]))
                    mag_dev = np.zeros(
                        (DF.shape[0], tmp.shape[0], tmp.shape[1]))
                xcor = file_dev_notpresent.variables[xvar][:].data
                ycor = file_dev_notpresent.variables[yvar][:].data
                first_run = False
            u = file_dev_notpresent.variables[uvar][:].data
            v = file_dev_notpresent.variables[vvar][:].data
            mag_nodev[ir, :] = np.sqrt(u**2 + v**2)
            u = file_dev_present.variables[uvar][:].data
            v = file_dev_present.variables[vvar][:].data
            mag_dev[ir, :] = np.sqrt(u**2 + v**2)

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
            xcor, ycor, mag_nodev, mag_dev = trim_zeros(
                xcor, ycor, mag_nodev, mag_dev)

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
        # ignor number and start sequentially from zero
        BC_probability['run_num'] = np.arange(0, mag_dev.shape[0])
        # assumes run_num in name is the return interval
        BC_probability["probability"] = 1/DF.run_num_dev.to_numpy()
        BC_probability["probability"] = BC_probability["probability"] / \
            BC_probability["probability"].sum()  # rescale to ensure = 1

    # ensure velocity is depth averaged for structured array [run_num, time, layer, x, y] and drop dimension
    if np.ndim(mag_nodev) == 5:
        mag_dev = np.nanmean(mag_dev, axis=2)
        mag_nodev = np.nanmean(mag_nodev, axis=2)

    # Calculate Stressor and Receptors
    if value_selection == 'MAX':
        mag_dev = np.nanmax(mag_dev, axis=1)  # max over time
        mag_nodev = np.nanmax(mag_nodev, axis=1)  # max over time
    elif value_selection == 'MEAN':
        mag_dev = np.nanmean(mag_dev, axis=1)  # mean over time
        mag_nodev = np.nanmean(mag_nodev, axis=1)  # mean over time
    elif value_selection == 'LAST':
        mag_dev = mag_dev[:, -1, :]  # last time step
        mag_nodev = mag_nodev[:, -1, :]  # last time step
    else:
        mag_dev = np.nanmax(mag_dev, axis=1)  # default to max over time
        mag_nodev = np.nanmax(mag_nodev, axis=1)  # default to max over time

    # initialize arrays
    if gridtype == 'structured':
        mag_combined_nodev = np.zeros(np.shape(mag_nodev[0, :, :]))
        mag_combined_dev = np.zeros(np.shape(mag_dev[0, :, :]))
    else:
        mag_combined_nodev = np.zeros(np.shape(mag_nodev)[-1])
        mag_combined_dev = np.zeros(np.shape(mag_dev)[-1])

    for run_number, prob in zip(BC_probability['run_num'].values,
                                BC_probability["probability"].values):

        mag_combined_nodev = mag_combined_nodev + \
            prob * mag_nodev[run_number, :]
        mag_combined_dev = mag_combined_dev + prob * mag_dev[run_number, :]

    mag_diff = mag_combined_dev - mag_combined_nodev
    velcrit = calc_receptor_array(receptor_filename, xcor, ycor, latlon=latlon)
    motility_nodev = mag_combined_nodev / velcrit
    motility_nodev = np.where(velcrit == 0, np.nan, motility_nodev)
    motility_dev = mag_combined_dev / velcrit
    motility_dev = np.where(velcrit == 0, np.nan, motility_dev)
    # Calculate risk metrics over all runs

    motility_diff = motility_dev - motility_nodev

    if gridtype == 'structured':
        motility_classification = classify_motility(
            motility_dev, motility_nodev)
        dx = np.nanmean(np.diff(xcor[:, 0]))
        dy = np.nanmean(np.diff(ycor[0, :]))
        rx = xcor
        ry = ycor
        # listOfFiles = [mag_combined_dev, mag_combined_nodev, mag_diff, motility_nodev,
        #                motility_dev, motility_diff, motility_classification, velcrit]
        dict_of_arrays = {'velocity_without_devices':mag_combined_nodev,
                        'velocity_with_devices': mag_combined_dev,
                        'velocity_difference': mag_diff,
                        'motility_without_devices': motility_nodev,
                        'motility_with_devices': motility_dev,
                        'motility_difference': motility_diff,
                        'motility_classified':motility_classification,
                        'critical_velocity':velcrit}        
    else:  # unstructured
        dxdy = estimate_grid_spacing(xcor, ycor, nsamples=100)
        dx = dxdy
        dy = dxdy
        rx, ry, mag_diff_struct = create_structured_array_from_unstructured(
            xcor, ycor, mag_diff, dxdy, flatness=0.2)
        _, _, mag_combined_dev_struct = create_structured_array_from_unstructured(
            xcor, ycor, mag_combined_dev, dxdy, flatness=0.2)
        _, _, mag_combined_nodev_struct = create_structured_array_from_unstructured(
            xcor, ycor, mag_combined_nodev, dxdy, flatness=0.2)
        if not ((receptor_filename is None) or (receptor_filename == "")):
            _, _, motility_nodev_struct = create_structured_array_from_unstructured(
                xcor, ycor, motility_nodev, dxdy, flatness=0.2)
            _, _, motility_dev_struct = create_structured_array_from_unstructured(
                xcor, ycor, motility_dev, dxdy, flatness=0.2)
            _, _, motility_diff_struct = create_structured_array_from_unstructured(
                xcor, ycor, motility_diff, dxdy, flatness=0.2)
            _, _, velcrit_struct = create_structured_array_from_unstructured(
                xcor, ycor, velcrit, dxdy, flatness=0.2)

        else:
            motility_nodev_struct = np.nan * mag_diff_struct
            motility_dev_struct = np.nan * mag_diff_struct
            motility_diff_struct = np.nan * mag_diff_struct
            velcrit_struct = np.nan * mag_diff_struct

        motility_classification = classify_motility(
            motility_dev_struct, motility_nodev_struct)
        motility_classification = np.where(
            np.isnan(mag_diff_struct), -100, motility_classification)
        # listOfFiles = [mag_combined_dev_struct, mag_combined_nodev_struct, mag_diff_struct, motility_nodev_struct,
        #                motility_dev_struct, motility_diff_struct, motility_classification, velcrit_struct]
        dict_of_arrays = {'velocity_magnitude_without_devices':mag_combined_nodev_struct,
                        'velocity_magnitude_with_devices': mag_combined_dev_struct,
                        'velocity_magnitude_difference': mag_diff_struct,
                        'motility_without_devices': motility_nodev_struct,
                        'motility_with_devices': motility_dev_struct,
                        'motility_difference': motility_diff_struct,
                        'motility_classified':motility_classification,
                        'critical_velocity':velcrit}
    return dict_of_arrays, rx, ry, dx, dy, gridtype


def run_velocity_stressor(
    dev_present_file,
    dev_notpresent_file,
    probabilities_file,
    crs,
    output_path,
    receptor_filename=None,
    secondary_constraint_filename=None,
):
    """
    creates geotiffs and area change statistics files for velocity change

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
    secondary_constraint_filename: str, optional
        File path to the secondary constraint file (*.tif). The default is None.

    Returns
    -------
    output_rasters : dict
        key = names of output rasters, val = full path to raster:
    """
    # numpy_arrays = [0] velocity_magnitude with devices
    #               [1] velocity_magnitude without devices
    #               [2] mag_diff
    #               [3] motility_nodev
    #               [4] motility_dev
    #               [5] motility_diff
    #               [6] motility_classification
    #               [7] receptor - vel_crit
    
    dict_of_arrays, rx, ry, dx, dy, gridtype = calculate_velocity_stressors(fpath_nodev=dev_notpresent_file,
                                                                          fpath_dev=dev_present_file,
                                                                          probabilities_file=probabilities_file,
                                                                          receptor_filename=receptor_filename,
                                                                          latlon=crs == 4326)

        # dict_of_arrays = {'velocity_magnitude_without_devices':mag_combined_nodev_struct,
        #                 'velocity_magnitude_with_devices': mag_combined_dev_struct,
        #                 'velocity_magnitude_difference': mag_diff_struct,
        #                 'motility_without_devices': motility_nodev_struct,
        #                 'motility_with_devices': motility_dev_struct,
        #                 'motility_difference': motility_diff_struct,
        #                 'motility_classified':motility_classification,
        #                 'critical_velocity':velcrit}

    if not ((receptor_filename is None) or (receptor_filename == "")):
        use_numpy_arrays = ['velocity_magnitude_without_devices',
                      'velocity_magnitude_with_devices',
                      'velocity_magnitude_difference',
                      'motility_without_devices',
                      'motility_with_devices',
                      'motility_difference',
                      'motility_classified',
                      'critical_velocity']
    else:
        use_numpy_arrays = ['velocity_magnitude_without_devices',
                      'velocity_magnitude_with_devices',
                      'velocity_magnitude_difference']

    if not ((secondary_constraint_filename is None) or (secondary_constraint_filename == "")):
        rrx, rry, constraint = secondary_constraint_geotiff_to_numpy(secondary_constraint_filename)
        dict_of_arrays['velocity_risk_layer'] = resample_structured_grid(rrx, rry, constraint, rx, ry, interpmethod='nearest')
        use_numpy_arrays.append('velocity_risk_layer.tif')

    numpy_array_names = [i + '.tif' for i in use_numpy_arrays]
    
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
        output_raster = None

    # Area calculations pull form rasters to ensure uniformity
    bin_layer(os.path.join(output_path, 'velocity_difference.tif'),
                receptor_filename=None,
                receptor_names=None,
                latlon=crs == 4326).to_csv(os.path.join(output_path, "velocity_difference.csv"), index=False)
    if not ((secondary_constraint_filename is None) or (secondary_constraint_filename == "")):
            bin_layer(os.path.join(output_path, 'velocity_difference.tif'),
                    receptor_filename=os.path.join(output_path, "velocity_risk_layer.tif"),
                    receptor_names=None,
                    limit_receptor_range=[0, np.inf],
                    latlon=crs == 4326).to_csv(os.path.join(output_path, "velocity_difference_at_velocity_risk_layer.csv"), index=False)
    if not ((receptor_filename is None) or (receptor_filename == "")):
        bin_layer(os.path.join(output_path, 'velocity_difference.tif'),
                    receptor_filename=os.path.join(output_path, 'critical_velocity.tif'),
                    receptor_names=None,
                    limit_receptor_range=[0, np.inf],
                    latlon=crs == 4326).to_csv(os.path.join(output_path, "velocity_difference_at_critical_velocity.csv"), index=False)
        
        bin_layer(os.path.join(output_path, 'motility_difference.tif'),
                    receptor_filename=None,
                    receptor_names=None,
                    limit_receptor_range=[0, np.inf],
                    latlon=crs == 4326).to_csv(os.path.join(output_path, "motility_difference.csv"), index=False)
            
        bin_layer(os.path.join(output_path, 'motility_difference.tif'),
                    receptor_filename=os.path.join(output_path, 'critical_velocity.tif'),
                    receptor_names=None,
                    limit_receptor_range=[0, np.inf],
                    latlon=crs == 4326).to_csv(os.path.join(output_path, "motility_difference_at_critical_velocity.csv"), index=False)
        
        classify_layer_area(os.path.join(output_path, "motility_classified.tif"),
                            at_values=[-3, -2, -1, 0, 1, 2, 3],
                            value_names=['New Deposition', 'Increased Deposition', 'Reduced Deposition',
                                            'No Change', 'Reduced Erosion', 'Increased Erosion', 'New Erosion'],
                            latlon=crs == 4326).to_csv(os.path.join(output_path, "motility_classified.csv"), index=False)
        
        classify_layer_area(os.path.join(output_path, "motility_classified.tif"),
                            receptor_filename=os.path.join(output_path, 'critical_velocity.tif'),
                            at_values=[-3, -2, -1, 0, 1, 2, 3],
                            value_names=['New Deposition', 'Increased Deposition', 'Reduced Deposition',
                                            'No Change', 'Reduced Erosion', 'Increased Erosion', 'New Erosion'],
                            limit_receptor_range=[0, np.inf],
                            latlon=crs == 4326).to_csv(os.path.join(output_path, "motility_classified_at_critical_velocity.csv"), index=False)
        
        if not ((secondary_constraint_filename is None) or (secondary_constraint_filename == "")):
            bin_layer(os.path.join(output_path, 'motility_difference.tif'),
                    receptor_filename=os.path.join(output_path, "velocity_risk_layer.tif"),
                    receptor_names=None,
                    limit_receptor_range=[0, np.inf],
                    latlon=crs == 4326).to_csv(os.path.join(output_path, "motility_difference_at_velocity_risk_layer.csv"), index=False)

            classify_layer_area_2nd_Constraint(raster_to_sample = os.path.join(output_path, "motility_classified.tif"),
                            secondary_constraint_filename=os.path.join(output_path, "velocity_risk_layer.tif"),
                            at_raster_values=[-3, -2, -1, 0, 1, 2, 3],
                            at_raster_value_names=['New Deposition', 'Increased Deposition', 'Reduced Deposition',
                                            'No Change', 'Reduced Erosion', 'Increased Erosion', 'New Erosion'],
                            limit_constraint_range=[0, np.inf],
                            latlon=crs == 4326).to_csv(os.path.join(output_path, "motility_classified_at_velocity_risk_layer.csv"), index=False)
    OUTPUT = {}
    for val in output_rasters:
        OUTPUT[os.path.basename(os.path.normpath(val)).split('.')[0]] = val            
    return OUTPUT
