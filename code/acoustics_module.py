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
    calculate_grid_sqarea_latlon2m
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
        # taucrit without a receptor
        #Assume the following grain sizes and conditions for typical beach sand (Nielsen, 1992 p.108)
        variable_array = np.ones(x.shape)
    return variable_array

def calculate_acoustic_stressors(fpath_dev, 
                                probabilities_file,
                                receptor_filename=None,
                                species_folder=None, #secondary constraint
                                latlon=True,
):
    paracousti_files = [os.path.join(fpath_dev, i) for i in os.listdir(fpath_dev) if i.endswith('.nc')]
    boundary_conditions = pd.read_csv(probabilities_file).set_index('Paracousti File').fillna(0)
    boundary_conditions['Paracousti Percent Occurance'] = 100 * (boundary_conditions['Paracousti Percent Occurance']/ boundary_conditions['Paracousti Percent Occurance'].sum())
    
    receptor = pd.read_csv(receptor_filename, index_col=0, header=None).T
    Threshold = receptor['Threshold (dB re 1uPa)'].astype(float).to_numpy().item()
    grid_res_species = receptor['species probability files grid resolution (km2)'].to_numpy.item() * 1.0e6 #converted to m2
    Averaging = receptor['Depth Averaging'].values.item()
    variable = receptor['Paracousti Variable'].values.item()

    for ic, paracousti_file in enumerate(paracousti_files):
        ds = Dataset(paracousti_file)
        spl = ds.variables[variable][:].data
        cords = ds.variables[variable].coordinates.split()
        X = ds.variables[cords[0]][:].data
        Y = ds.variables[cords[1]][:].data
        if X.shape[0] != spl.shape[0]:
            spl = np.transpose(spl, (1, 2, 0))
        if ic==0:
            xunits = ds.variables['XCOR'].units
            if 'degrees' in xunits:
                XCOR = np.where(X<0, X+360, X)
            else:
                XCOR = X
            YCOR = Y
            SPL = np.zeros((len(paracousti_files), np.shape(spl)[0], np.shape(spl)[1], np.shape(spl)[2]))
        SPL[ic,:] = spl

    if Averaging == 'DepthMax':
        SPL = np.nanmax(SPL, axis=3)
    elif Averaging == 'DepthAverage':
        SPL = np.nanmean(SPL, axis=3)
    elif Averaging == 'Bottom':
        SPL = SPL[:,:,-1]
    elif Averaging == 'Top':
        SPL = SPL[:,:,0]
    else:
        SPL = np.nanmax(SPL, axis=3)

    for ic, file in enumerate(paracousti_files):
        if ic==0:
            stressor = np.zeros(XCOR.shape)
            species_percent_occurance = np.zeros(XCOR.shape)
            species_density = np.zeros(XCOR.shape)
            threshold_exceeded = np.zeros(XCOR.shape)
            percent_impacted = np.zeros(XCOR.shape)
            density_impacted = np.zeros(XCOR.shape)
            percent_impacted_scaled = np.zeros(XCOR.shape)
            density_impacted_scaled = np.zeros(XCOR.shape)

        probability = boundary_conditions.loc[os.path.basename(file)]['Paracousti Percent Occurance'] / 100
        species_percent_filename = boundary_conditions.loc[os.path.basename(paracousti_file)]['Species Percent Occurance File']
        species_density_filename = boundary_conditions.loc[os.path.basename(paracousti_file)]['Species Density File']
        
        rx, ry, spl = redefine_structured_grid(XCOR, YCOR, SPL[ic,:]) #paracousti files might not have regular grid spacing.
        stressor = stressor + probability * spl
        _, _, square_area = calculate_grid_sqarea_latlon2m(rx, ry) #TODO this only applies to lat/lon, add option for already in UTM
        square_area = np.nanmean(square_area) # square area of each grid cell
        ratio = square_area / grid_res_species # ratio of grid cell to species averaged, now prob/density per each grid cell
        parray = create_whale_array(os.path.join(species_percent_filename), rx, ry, variable='percent', latlon=True)
        darray = create_whale_array(os.path.join(species_density_filename), rx, ry, variable='density', latlon=True)
        parray_scaled = parray * ratio
        darray_scaled = darray  * ratio

        species_percent_occurance = species_percent_occurance + probability * parray
        species_density = species_density + probability * darray

        species_percent_occurance_scaled = species_percent_occurance + probability * parray_scaled
        species_density_scaled = species_density + probability * darray_scaled

        threshold_mask = spl>Threshold
        threshold_exceeded[threshold_mask] += probability*100#np.flatnonzero(threshold_mask, threshold_time_exceeded+probability, threshold_exceeded)
        percent_impacted[threshold_mask] += probability * parray[threshold_mask]
        density_impacted[threshold_mask] += probability * darray[threshold_mask]
        percent_impacted_scaled[threshold_mask] += probability * parray_scaled[threshold_mask]
        density_impacted_scaled[threshold_mask] += probability * darray_scaled[threshold_mask]

    Threshold_Exceeded = np.where(stressor > Threshold, 1, 0)
    Percent_Impacted = np.where(Threshold_Exceeded == 1, species_percent_occurance, 0)
    Density_Impacted = np.where(Threshold_Exceeded == 1, species_density, 0)
    Percent_Impacted_scaled = np.where(Threshold_Exceeded == 1, species_percent_occurance_scaled, 0)
    Density_Impacted_scaled = np.where(Threshold_Exceeded == 1, species_density_scaled, 0)

    listOfFiles = [stressor, Threshold_Exceeded, Percent_Impacted, Density_Impacted, threshold_exceeded, percent_impacted, density_impacted, Percent_Impacted_scaled, Density_Impacted_scaled, percent_impacted_scaled, density_impacted_scaled]
    dx = np.nanmean(np.diff(rx[:,0]))
    dy = np.nanmean(np.diff(ry[0,:]))
    return listOfFiles, rx, ry, dx, dy

def run_acoustics_stressor(
    dev_present_file,
    bc_file,
    crs,
    output_path,
    receptor_filename=None,
    species_folder=None
):
    
    numpy_arrays, rx, ry, dx, dy = calculate_acoustic_stressors(fpath_dev=dev_present_file, 
                                probabilities_file=bc_file,
                                receptor_filename=receptor_filename,
                                species_folder=species_folder, #secondary constraint
                                latlon = crs==4326)
    
    #numpy_arrays = [0] stressor
    #               [1] Threshold_Exceeded
    #               [2] Percent_Impacted
    #               [3] Density_Impacted
    #               [4] threshold_exceeded
    #               [5] percent_impacted
    #               [6] density_impacted   
    #               [7] Percent_Impacted_scaled
    #               [8] Density_Impacted_scaled
    #               [9] percent_impacted_scaled
    #               [10] density_impacted_scaled

    if not((receptor_filename is None) or (receptor_filename == "")):
        numpy_array_names = ['calculated_stressor.tif',
                             'threshold_exceeded_receptor'
                            'species_percent_impacted_averaged_stresser.tif',
                            'species_density_impacted_averaged_stresser.tif',
                            'threshold_exceeded_summation.tif',
                            'species_percent_impacted_summation.tif',
                            'species_density_impacted_summation.tif',
                            'scaled_species_percent_impacted_averaged_stresser.tif',
                            'scaled_species_density_impacted_averaged_stresser.tif',
                            'scaled_species_percent_impacted_summation.tif',
                            'scaled_species_density_impacted_summation.tif']
        use_numpy_arrays = [numpy_arrays[0], numpy_arrays[1], numpy_arrays[2], numpy_arrays[3], numpy_arrays[4], numpy_arrays[5], numpy_arrays[6], numpy_arrays[7], numpy_arrays[8], numpy_arrays[9], numpy_arrays[10]]
    else:
        numpy_array_names = ['calculated_stressor.tif']
        use_numpy_arrays = [numpy_arrays[0]]
    
    output_rasters = []
    for array_name, numpy_array in zip(numpy_array_names, use_numpy_arrays):

        numpy_array = np.flip(np.transpose(numpy_array), axis=0)


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

        #TODO add area impacted and/or summation of not zero
    return output_rasters