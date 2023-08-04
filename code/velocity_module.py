#!/usr/bin/python

# Oregon Coast Delft3D WEC Difference Plotting

# Plot normalized comparison of simulations with WECs and without WECS for
# user selected variable for all boundary conditions

# Usage:
# python delft_wecs_diff_all_bcs.py

# Output: # of BCs figures saved to run_dir directory

import glob
import os

import numpy as np
import pandas as pd
from matplotlib.tri import LinearTriInterpolator, TriAnalyzer, Triangulation
from scipy.interpolate import griddata
from netCDF4 import Dataset
from osgeo import gdal, osr
from .readnetcdf_createraster import (
    create_raster,
    numpy_array_to_raster
)
from .stressor_utils import (
    estimate_grid_spacing,
    create_structured_array_from_unstructured,
    calc_receptor_array,
    trim_zeros
)

def calculate_receptor_change_percentage(receptor_array, data_diff):
    #gdal version
    # data_diff = np.transpose(data_diff)
    # data = gdal.Open(receptor_filename)
    # img = data.GetRasterBand(1)
    # receptor_array = img.ReadAsArray()
    # transpose to be in same orientation as NetCDF
    receptor_array[receptor_array < 0] = np.nan

    pct_mobility = {'Receptor_Value': [],
                        'Increased Deposition': [],
                        'Reduced Deposition': [],
                        'Reduced Mobility': [],
                        'Increased Mobility': [],
                        'No Change':[]}
    
    for unique_val in [i for i in np.unique(receptor_array) if ~np.isnan(i)]:
        mask = receptor_array==unique_val
        data_at_val = np.where(mask, data_diff, np.nan)
        data_at_val = data_at_val.flatten()
        data_at_val = data_at_val[~np.isnan(data_at_val)]
        ncells = data_at_val.size
        pct_mobility['Receptor_Value'].append(unique_val)
        pct_mobility['Increased Deposition'].append(100 * np.size(np.flatnonzero(data_at_val==-2))/ncells)
        pct_mobility['Reduced Deposition'].append(100 * np.size(np.flatnonzero(data_at_val==-1))/ncells)
        pct_mobility['Reduced Mobility'].append(100 * np.size(np.flatnonzero(data_at_val==1))/ncells)
        pct_mobility['Increased Mobility'].append(100 * np.size(np.flatnonzero(data_at_val==2))/ncells)
        pct_mobility['No Change'].append(100 * np.size(np.flatnonzero(data_at_val==0))/ncells)

        # print(f" Receptor Value = {unique_val}um | decrease = {pct_decrease}% | increase = {pct_increase}% | no change = {pct_nochange}%")
    DF = pd.DataFrame(pct_mobility)
    DF = DF.set_index('Receptor_Value')
    return DF
    # DF.to_csv(os.path.join(ofpath, 'receptor_percent_change.csv'))

def classify_mobility(mobility_parameter_dev, mobility_parameter_nodev):
    mobility_classification = np.zeros(mobility_parameter_dev.shape)
    #Reduced Erosion (Tw<Tb) & (Tw-Tb)>1
    mobility_classification = np.where(((mobility_parameter_dev < mobility_parameter_nodev) & (mobility_parameter_nodev>=1)), 1, mobility_classification)
    #Increased Erosion (Tw>Tb) & (Tw-Tb)>1
    mobility_classification = np.where(((mobility_parameter_dev > mobility_parameter_nodev) & (mobility_parameter_nodev>=1)), 2, mobility_classification)
    #Reduced Deposition (Tw>Tb) & (Tw-Tb)<1
    mobility_classification = np.where(((mobility_parameter_dev > mobility_parameter_nodev) & (mobility_parameter_nodev<1)), -1, mobility_classification)
    #Increased Deposition (Tw>Tb) & (Tw-Tb)>1
    mobility_classification = np.where(((mobility_parameter_dev < mobility_parameter_nodev) & (mobility_parameter_nodev<1)), -2, mobility_classification)
    #NoChange = 0
    return mobility_classification

def check_grid_define_vars(dataset):
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
    files_nodev = [i for i in os.listdir(fpath_nodev) if i.endswith('.nc')]
    files_dev = [i for i in os.listdir(fpath_dev) if i.endswith('.nc')]

    # Load and sort files
    if len(files_nodev) == 1 & len(files_dev) == 1: 
        #asumes a concatonated files with shape
        #[run_order, time, rows, cols]

        file_dev_present = Dataset(os.path.join(fpath_dev, files_dev[0]))
        gridtype, xvar, yvar, uvar, vvar = check_grid_define_vars(file_dev_present)
        xcor = file_dev_present.variables[xvar][:].data
        ycor = file_dev_present.variables[yvar][:].data
        u = file_dev_present.variables[uvar][:].data
        v = file_dev_present.variables[vvar][:].data
        mag_dev = np.sqrt(u**2 + v**2)

        # close the device prsent file
        file_dev_present.close()

        file_dev_notpresent = Dataset(os.path.join(fpath_nodev, files_nodev[0]))
        u = file_dev_notpresent.variables[uvar][:].data
        v = file_dev_notpresent.variables[vvar][:].data
        mag_nodev = np.sqrt(u**2 + v**2)
        # close the device not present file
        file_dev_notpresent.close()
        
        # if mag_dev.shape[0] != mag_nodev.shape[0]:
        #     raise Exception(f"Number of device runs ({mag_dev.shape[0]}) must be the same as no device runs ({mag_nodev.shape[0]}).") 

    elif len(files_nodev) == len(files_dev): #same number of files, file name must be formatted with either run number or return interval
        #asumes each run is separate with the some_name_RunNum_map.nc, where run number comes at the last underscore before _map.nc
        runorder_nodev = np.zeros((len(files_nodev)))
        for ic, file in enumerate(files_nodev):
            runorder_nodev[ic] = int(file.split('.')[0].split('_')[-2])
        runorder_dev = np.zeros((len(files_dev)))
        for ic, file in enumerate(files_dev):
            runorder_dev[ic] = int(file.split('.')[0].split('_')[-2])

        #ensure run oder for nodev matches dev files
        if np.any(runorder_nodev != runorder_dev):
            adjust_dev_order = []
            for ri in runorder_nodev:
                adjust_dev_order = np.append(adjust_dev_order, np.flatnonzero(runorder_dev == ri))
            files_dev = [files_dev[int(i)] for i in adjust_dev_order]
            runorder_dev = [runorder_dev[int(i)] for i in adjust_dev_order]
        DF = pd.DataFrame({'files_nodev':files_nodev, 
                    'run_order_nodev':runorder_nodev,
                    'files_dev':files_dev,
                    'run_order_dev':runorder_dev})
        DF = DF.sort_values(by='run_order_dev')

        first_run = True
        ir = 0
        for _, row in DF.iterrows():
            file_dev_notpresent = Dataset(os.path.join(fpath_nodev, row.files_nodev))
            file_dev_present = Dataset(os.path.join(fpath_dev, row.files_dev))

            gridtype, xvar, yvar, uvar, vvar = check_grid_define_vars(file_dev_present)

            if first_run:
                tmp = file_dev_notpresent.variables[uvar][:].data
                if gridtype=='structured':
                    mag_nodev = np.zeros((DF.shape[0], tmp.shape[0], tmp.shape[1], tmp.shape[2], tmp.shape[3]))
                    mag_dev = np.zeros((DF.shape[0], tmp.shape[0], tmp.shape[1], tmp.shape[2], tmp.shape))
                else:
                    mag_nodev = np.zeros((DF.shape[0], tmp.shape[0], tmp.shape[1]))
                    mag_dev = np.zeros((DF.shape[0], tmp.shape[0], tmp.shape[1]))
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
        raise Exception(f"Number of device runs ({len(files_dev)}) must be the same as no device runs ({len(files_nodev)}).") 
    # Finished loading and sorting files

    if (gridtype == 'structured'):
        if (xcor[0,0]==0) & (xcor[-1,0]==0):
        #at least for some runs the boundary has 0 coordinates. Check and fix.
            xcor, ycor, mag_nodev, mag_dev = trim_zeros(xcor, ycor, mag_nodev, mag_dev)

    if not(probabilities_file == ""):
        # Load BC file with probabilities and find appropriate probability
        BC_probability = pd.read_csv(probabilities_file, delimiter=",")
        BC_probability['run order'] = BC_probability['run order']-1
        BC_probability = BC_probability.sort_values(by='run order')
        BC_probability["probability"]= BC_probability["% of yr"].values/100
        # BC_probability
        if 'Exclude' in BC_probability.columns:
            BC_probability = BC_probability[~((BC_probability['Exclude'] == 'x') | (BC_probability['Exclude'] == 'X'))]
    else: #assume run_order in file name is return interval
        BC_probability = pd.DataFrame()
        BC_probability['run order'] = np.arange(0,mag_dev.shape[0]) #ignor number and start sequentially from zero
        BC_probability["probability"] = 1/DF.run_order_dev.to_numpy() #assumes run order in name is the return interval

    #ensure velocity is depth averaged for structured array [run order, time, layer, x, y] and drop dimension
    if np.ndim(mag_nodev) == 5:
        mag_dev = np.nanmean(mag_dev, axis=2)
        mag_nodev = np.nanmean(mag_nodev, axis=2)

    # Calculate Stressor and Receptors
    # data_dev_max = np.amax(data_dev, axis=1, keepdims=True) #look at maximum shear stress difference change
    if value_selection == 'MAX':
        mag_dev = np.nanmax(mag_dev, axis=1) #max over time
        mag_nodev = np.nanmax(mag_nodev, axis=1) #max over time
    elif value_selection == 'MEAN':
        mag_dev = np.nanmean(mag_dev, axis=1) #max over time
        mag_nodev = np.nanmean(mag_nodev, axis=1) #max over time
    elif value_selection == 'LAST':
        mag_dev = mag_dev[:, -1, :] #max over time
        mag_nodev = mag_nodev[:, -1, :] #max over time
    else:
        mag_dev = np.nanmax(mag_dev, axis=1) #max over time
        mag_nodev = np.nanmax(mag_nodev, axis=1) #max over time

    #initialize arrays
    if gridtype == 'structured':
        mag_combined_nodev = np.zeros(np.shape(mag_nodev[0, :, :]))
        mag_combined_dev = np.zeros(np.shape(mag_dev[0, :, :]))
    else:
        mag_combined_nodev = np.zeros(np.shape(mag_nodev)[-1])
        mag_combined_dev = np.zeros(np.shape(mag_dev)[-1])

    for run_number, prob in zip(BC_probability['run order'].values,
                                BC_probability["probability"].values):
            
        mag_combined_nodev = mag_combined_nodev + prob * mag_nodev[run_number,:] 
        mag_combined_dev = mag_combined_dev + prob * mag_dev[run_number,:] 

    mag_diff = mag_combined_dev - mag_combined_nodev
    velcrit = calc_receptor_array(receptor_filename, xcor, ycor, latlon=latlon)
    mobility_nodev = mag_combined_nodev / velcrit
    mobility_nodev = np.where(velcrit==0, 0, mobility_nodev)
    mobility_dev = mag_combined_dev / velcrit
    mobility_dev = np.where(velcrit==0, 0, mobility_dev)
    # Calculate risk metrics over all runs

    mobility_diff = mobility_dev - mobility_nodev


    if gridtype=='structured':
        mobility_classification = classify_mobility(mobility_dev, mobility_nodev)
        dx = np.nanmean(np.diff(xcor[:,0]))
        dy = np.nanmean(np.diff(ycor[0,:]))
        rx = xcor
        ry = ycor
        DF_classified = calculate_receptor_change_percentage(velcrit, mobility_classification)
        listOfFiles = [mag_diff, mobility_nodev, mobility_dev, mobility_diff, mobility_classification, velcrit]
    else: # unstructured
        dxdy = estimate_grid_spacing(xcor,ycor, nsamples=100)
        dx = dxdy
        dy = dxdy
        rx, ry, mag_diff_struct = create_structured_array_from_unstructured(xcor, ycor, mag_diff, dxdy, flatness=0.2)
        if not((receptor_filename is None) or (receptor_filename == "")):
            _, _, mobility_nodev_struct = create_structured_array_from_unstructured(xcor, ycor, mobility_nodev, dxdy, flatness=0.2)
            _, _, mobility_dev_struct = create_structured_array_from_unstructured(xcor, ycor, mobility_dev, dxdy, flatness=0.2)
            _, _, mobility_diff_struct = create_structured_array_from_unstructured(xcor, ycor, mobility_diff, dxdy, flatness=0.2)
            _, _, velcrit_struct = create_structured_array_from_unstructured(xcor, ycor, velcrit, dxdy, flatness=0.2)
        else:
            mobility_nodev_struct = np.nan * mag_diff_struct
            mobility_dev_struct = np.nan * mag_diff_struct
            mobility_diff_struct = np.nan * mag_diff_struct
            velcrit_struct = np.nan * mag_diff_struct
        mobility_classification = classify_mobility(mobility_dev_struct, mobility_nodev_struct)
        DF_classified = calculate_receptor_change_percentage(velcrit_struct, mobility_classification)
        listOfFiles = [mag_diff_struct, mobility_nodev_struct, mobility_dev_struct, mobility_diff_struct, mobility_classification, velcrit]

    return listOfFiles, rx, ry, dx, dy, DF_classified, gridtype

def run_velocity_stressor(
    dev_present_file,
    dev_notpresent_file,
    bc_file,
    crs,
    output_path,
    receptor_filename=None
):
    
    numpy_arrays, rx, ry, dx, dy, DF_classified, gridtype = calculate_velocity_stressors(fpath_nodev=dev_notpresent_file, 
                                                fpath_dev=dev_present_file, 
                                                probabilities_file=bc_file,
                                                receptor_filename=receptor_filename,
                                                latlon= crs==4326)
    #numpy_arrays = [0] mag_diff
    #               [1] mobility_nodev
    #               [2] mobility_dev
    #               [3] mobility_diff
    #               [4] mobility_classification    
    if not((receptor_filename is None) or (receptor_filename == "")):
        numpy_array_names = ['calculated_stressor.tif',
                            'calculated_stressor_with_receptor.tif',
                            'calculated_stressor_reclassified.tif']
        use_numpy_arrays = [numpy_arrays[0], numpy_arrays[3], numpy_arrays[4]]
        DF_classified.to_csv(os.path.join(output_path, 'receptor_percent_change.csv'))
    else:
        numpy_array_names = ['calculated_stressor.tif']
        use_numpy_arrays = [numpy_arrays[0]]
    
    output_rasters = []
    for array_name, numpy_array in zip(numpy_array_names, use_numpy_arrays):

        if gridtype=='structured':
            numpy_array = np.flip(np.transpose(numpy_array), axis=0)
        else:
            numpy_array = np.flip(numpy_array, axis=0)

        cell_resolution = [dx, dy]
        if crs == 4326:
            bounds = [rx.min()-360 - dx/2, ry.max() - dy/2]
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
    # if not((receptor_filename is None) or (receptor_filename == "")):
    #     # print('calculating percentages')
    #     # print(output_path)
    #     calculate_receptor_change_percentage(receptor_filename=receptor_filename, data_diff=numpy_arrays[-1], ofpath=os.path.dirname(output_path))


    return output_rasters