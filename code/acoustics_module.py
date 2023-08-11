import os
from scipy.interpolate import griddata
from netCDF4 import Dataset
import pandas as pd
from osgeo import gdal, osr
from .stressor_utils import (
    redefine_structured_grid
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
                                latlon=True
):
    paracousti_files = [os.path.join(fpath_dev, i) for i in os.listdir(fpath_dev) if i.endswith('.nc')]
    boundary_conditions = pd.read_csv(probabilities_file).set_index('Paracousti File').fillna(0)
    boundary_conditions['Paracousti Percent Occurance'] = 100 * (boundary_conditions['Paracousti Percent Occurance']/ boundary_conditions['Paracousti Percent Occurance'].sum())
    
    species_thresholds = pd.read_csv(receptor_filename, index_col=0, header=None).T
    Threshold = species_thresholds['Threshold (dB re 1uPa)'].astype(float).to_numpy().item()
    Averaging = species_thresholds['Depth Averaging'].values.item()
    variable = species_thresholds['Paracousti Variable'].values.item()

    for ic, paracousti_file in enumerate(paracousti_files):
        ds = Dataset(paracousti_file)
        spl = ds.variables[variable][:].data
        cords = ds.variables[variable].coordinates.split()
        X = ds.variables[cords[0]][:].data
        Y = ds.variables[cords[1]][:].data
        import numpy as np
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


    threshold_scale = 0.5

    stressor = np.zeros(XCOR.shape)
    species_percent_occurance = np.zeros(XCOR.shape)
    species_density = np.zeros(XCOR.shape)
    threshold_time_exceeded = np.zeros(XCOR.shape)
    percent_impacted = np.zeros(XCOR.shape)
    density_impacted = np.zeros(XCOR.shape)
    for ic, file in enumerate(paracousti_files):
        paracousti_file = os.path.basename(file)
        probability = boundary_conditions.loc[os.path.basename(paracousti_file)]['Paracousti Percent Occurance'] / 100
        stressor = stressor + probability * SPL[ic,:]

        species_percent_filename = boundary_conditions.loc[os.path.basename(paracousti_file)]['Species Percent Occurance File']
        species_density_filename = boundary_conditions.loc[os.path.basename(paracousti_file)]['Species Density File']
        parray = create_whale_array(os.path.join(species_percent_filename), XCOR, YCOR, variable='percent', latlon=True)
        darray = create_whale_array(os.path.join(species_density_filename), XCOR, YCOR, variable='density', latlon=True)

        species_percent_occurance = species_percent_occurance + probability * parray
        species_density = species_density + probability * darray

        threshold_mask = SPL[ic,:]>Threshold*threshold_scale
        threshold_time_exceeded[threshold_mask] += probability*100#np.flatnonzero(threshold_mask, threshold_time_exceeded+probability, threshold_exceeded)
        percent_impacted[threshold_mask] += probability * parray[threshold_mask]
        density_impacted[threshold_mask] += probability * darray[threshold_mask]

    Threshold_Exceeded = np.where(stressor > Threshold*threshold_scale, 1, 0)
    Percent_Impacted = np.where(Threshold_Exceeded == 1, species_percent_occurance, 0)
    Density_Impacted = np.where(Threshold_Exceeded == 1, species_density, 0)

    XX, YY, stressor = redefine_structured_grid(XCOR, YCOR, stressor)
    _, _, Percent_Impacted = redefine_structured_grid(XCOR, YCOR, Percent_Impacted)
    _, _, Density_Impacted = redefine_structured_grid(XCOR, YCOR, Density_Impacted)
    _, _, threshold_time_exceeded = redefine_structured_grid(XCOR, YCOR, threshold_time_exceeded)
    _, _, percent_impacted = redefine_structured_grid(XCOR, YCOR, percent_impacted)
    _, _, density_impacted = redefine_structured_grid(XCOR, YCOR, density_impacted)
    listOfFiles = [stressor, Percent_Impacted, Density_Impacted, threshold_time_exceeded, percent_impacted, density_impacted]
    dx = np.nanmean(np.diff(XX[:,0]))
    dy = np.nanmean(np.diff(YY[0,:]))
    return listOfFiles, XX, YY, dx, dy