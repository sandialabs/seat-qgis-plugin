import os
from scipy.interpolate import griddata
from netCDF4 import Dataset
import pandas as pd
from osgeo import gdal, osr
import numpy as np
from .stressor_utils import (
    redefine_structured_grid,
    create_raster,
    numpy_array_to_raster,
    calculate_cell_area, 
    resample_structured_grid,
    bin_layer
)

def create_whale_array(species_filename, x, y, variable='percent', latlon=False):
    # if ((receptor_filename is not None) or (not receptor_filename == "")):
    if not((species_filename is None) or (species_filename == "")):
        if species_filename.endswith('.tif'):
            data = gdal.Open(species_filename)
            img = data.GetRasterBand(1)
            receptor_array = img.ReadAsArray()
            receptor_array[receptor_array < 0] = 0
            (upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = data.GetGeoTransform()
            cols = data.RasterXSize
            rows = data.RasterYSize
            r_rows = np.arange(rows) * y_size + upper_left_y + (y_size / 2)
            r_cols = np.arange(cols) * x_size + upper_left_x + (x_size / 2)
            if latlon==True:
                r_cols = np.where(r_cols<0, r_cols+360, r_cols)
            x_grid, y_grid = np.meshgrid(r_cols, r_rows)
            variable_array = griddata((x_grid.flatten(), y_grid.flatten()), receptor_array.flatten(), (x,y), method='nearest', fill_value=0)

        elif species_filename.endswith('.csv'):
            df = pd.read_csv(species_filename) 
            columns_keep = ['latitude', 'longitude', variable]
            df = df[columns_keep]
            variable_array = griddata((df.longitude.to_numpy(), df.latitude.to_numpy()), df[variable].to_numpy(), (x,y), method='nearest', fill_value=0)
        else:
            raise Exception("Invalid File Type. Must be of type .tif or .csv")
    else:
        variable_array = np.zeros(x.shape)
    return variable_array

def calculate_acoustic_stressors(fpath_dev,
                                probabilities_file,
                                fpath_nodev=None,
                                receptor_filename=None,
                                species_folder=None, #secondary constraint
                                latlon=True,
):
    paracousti_files = [os.path.join(fpath_dev, i) for i in os.listdir(fpath_dev) if i.endswith('.nc')]
    boundary_conditions = pd.read_csv(probabilities_file).set_index('Paracousti File').fillna(0)
    boundary_conditions['Paracousti Percent Occurance'] = 100 * (boundary_conditions['Paracousti Percent Occurance']/ boundary_conditions['Paracousti Percent Occurance'].sum())
    
    receptor = pd.read_csv(receptor_filename, index_col=0, header=None).T
    Threshold = receptor['Threshold (dB re 1uPa)'].astype(float).to_numpy().item()
    if not((receptor['species file averaged area (km2)'].values is None) or (receptor['species file averaged area (km2)'].values == "")):
        grid_res_species = receptor['species file averaged area (km2)'].astype(float).to_numpy().item() * 1.0e6 #converted to m2
    else:
        grid_res_species = 0.0
    Averaging = receptor['Depth Averaging'].values.item()
    variable = receptor['Paracousti Variable'].values.item()

    for ic, paracousti_file in enumerate(paracousti_files):
        ds = Dataset(paracousti_file)
        acoust_var = ds.variables[variable][:].data
        cords = ds.variables[variable].coordinates.split()
        X = ds.variables[cords[0]][:].data
        Y = ds.variables[cords[1]][:].data
        if X.shape[0] != acoust_var.shape[0]:
            acoust_var = np.transpose(acoust_var, (1, 2, 0))
        if ic==0:
            xunits = ds.variables['XCOR'].units
            if 'degrees' in xunits:
                latlon = True
                XCOR = np.where(X<0, X+360, X)
            else:
                XCOR = X
            YCOR = Y
            ACOUST_VAR = np.zeros((len(paracousti_files), np.shape(acoust_var)[0], np.shape(acoust_var)[1], np.shape(acoust_var)[2]))
        ACOUST_VAR[ic,:] = acoust_var

    if not((fpath_nodev is None) or (fpath_nodev == "")): #Assumes same grid as paracousti_files
        baseline_files = [os.path.join(fpath_nodev, i) for i in os.listdir(fpath_nodev) if i.endswith('.nc')]
        for ic, baseline_file in enumerate(baseline_files):
            ds = Dataset(baseline_file)
            baseline = ds.variables[variable][:].data
            cords = ds.variables[variable].coordinates.split()
            if ds.variables[cords[0]][:].data.shape[0] != baseline.shape[0]:
                baseline = np.transpose(baseline, (1, 2, 0))
            if ic==0:
                Baseline = np.zeros((len(baseline_files), np.shape(baseline)[0], np.shape(baseline)[1], np.shape(baseline)[2]))
            Baseline[ic,:] = baseline
    else:
        Baseline = np.zeros(ACOUST_VAR.shape)


    if Averaging == 'DepthMax':
        ACOUST_VAR = np.nanmax(ACOUST_VAR, axis=3)
        Baseline = np.nanmax(Baseline, axis=3)
    elif Averaging == 'DepthAverage':
        ACOUST_VAR = np.nanmean(ACOUST_VAR, axis=3)
        Baseline = np.nanmean(Baseline, axis=3)
    elif Averaging == 'Bottom':
        ACOUST_VAR = ACOUST_VAR[:,:,-1]
        Baseline = Baseline[:,:,-1]
    elif Averaging == 'Top':
        ACOUST_VAR = ACOUST_VAR[:,:,0]
        Baseline = Baseline[:,:,0]
    else:
        ACOUST_VAR = np.nanmax(ACOUST_VAR, axis=3)
        Baseline = np.nanmax(Baseline, axis=3)

    for ic, file in enumerate(paracousti_files):
        rx, ry, acoust_var = redefine_structured_grid(XCOR, YCOR, ACOUST_VAR[ic,:]) #paracousti files might not have regular grid spacing.
        baseline = resample_structured_grid(XCOR, YCOR,  Baseline[ic,:], rx, ry)
        
        if ic==0:
            PARACOUSTI = np.zeros(rx.shape)
            stressor = np.zeros(rx.shape)
            # species_percent_occurance = np.zeros(rx.shape)
            # species_density = np.zeros(rx.shape)
            threshold_exceeded = np.zeros(rx.shape)
            percent = np.zeros(rx.shape)
            density = np.zeros(rx.shape)
            percent_scaled = np.zeros(rx.shape)
            density_scaled = np.zeros(rx.shape)

        probability = boundary_conditions.loc[os.path.basename(file)]['Paracousti Percent Occurance'] / 100
        species_percent_filename = boundary_conditions.loc[os.path.basename(paracousti_file)]['Species Percent Occurance File']
        species_density_filename = boundary_conditions.loc[os.path.basename(paracousti_file)]['Species Density File']
        

        PARACOUSTI = PARACOUSTI + probability * acoust_var
        stressor = stressor + probability * (acoust_var - baseline)
        # if latlon == True:
        _, _, square_area = calculate_cell_area(rx, ry, latlon == True) 
        square_area = np.nanmean(square_area) # square area of each grid cell
        # else:
        #     square_area = np.nanmean(np.diff(rx[0,:])) * np.nanmean(np.diff(ry[:,0]))
        if grid_res_species != 0:
            ratio = square_area / grid_res_species # ratio of grid cell to species averaged, now prob/density per each grid cell
        else:
            ratio = 1
        parray = create_whale_array(os.path.join(species_folder, species_percent_filename), rx, ry, variable='percent', latlon=True)
        darray = create_whale_array(os.path.join(species_folder, species_density_filename), rx, ry, variable='density', latlon=True)
        parray_scaled = parray * ratio
        darray_scaled = darray  * ratio

        threshold_mask = acoust_var>Threshold
        threshold_exceeded[threshold_mask] += probability*100
        percent[threshold_mask] += probability * parray[threshold_mask]
        density[threshold_mask] += probability * darray[threshold_mask]
        percent_scaled[threshold_mask] += probability * parray_scaled[threshold_mask]
        density_scaled[threshold_mask] += probability * darray_scaled[threshold_mask]

   

    listOfFiles = [PARACOUSTI, stressor, threshold_exceeded, percent, density, percent_scaled, density_scaled]
    dx = np.nanmean(np.diff(rx[0,:]))
    dy = np.nanmean(np.diff(ry[:,0]))
    return listOfFiles, rx, ry, dx, dy

def run_acoustics_stressor(
    dev_present_file,
    dev_notpresent_file,
    bc_file,
    crs,
    output_path,
    receptor_filename=None,
    species_folder=None
):
    
    numpy_arrays, rx, ry, dx, dy = calculate_acoustic_stressors(fpath_dev=dev_present_file, 
                                probabilities_file=bc_file,
                                fpath_nodev=dev_notpresent_file,
                                receptor_filename=receptor_filename,
                                species_folder=species_folder, #secondary constraint
                                latlon = crs==4326)
    
    #numpy_arrays = [0] PARACOUSTI
    #               [1] stressor
    #               [2] threshold_exceeded
    #               [3] percent
    #               [4] density
    #               [5] percent_scaled
    #               [6] density_scaled
    
    if not((receptor_filename is None) or (receptor_filename == "")):
        numpy_array_names = ['calculated_paracousti.tif',
                            'calculated_stressor.tif',
                            'threshold_exceeded_receptor.tif',
                            'species_percent.tif',
                            'species_density.tif',
                            'species_percent_scaled.tif',
                            'species_density_scaled.tif']
        use_numpy_arrays = [numpy_arrays[0], numpy_arrays[1], numpy_arrays[2], numpy_arrays[3], numpy_arrays[4], numpy_arrays[5], numpy_arrays[6]]
    else:
        numpy_array_names = ['calculated_paracousti.tif', 'calculated_stressor.tif']
        use_numpy_arrays = [numpy_arrays[0], numpy_arrays[1]]
    
    output_rasters = []
    for array_name, numpy_array in zip(numpy_array_names, use_numpy_arrays):

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
            eType=gdal.GDT_Float64,
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

    bin_layer(rx, ry, numpy_arrays[1], latlon=True).to_csv(os.path.join(output_path, "calculated_stressor.csv"), index=False)


    return output_rasters