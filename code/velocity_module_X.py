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

# these imports currently don't work :(
# from qgis.core import *
# import qgis.utils

def critical_shear_stress(D_meters, rhow=1024, nu=1e-6, s=2.65, g=9.81):
    # D_meters = grain size in meters, can be array
    # rhow = density of water in kg/m3
    # nu = kinematic viscosity of water
    # s = specific gravity of sediment
    # g = acceleratin due to gravity
    Dstar = ((g * (s-1))/ nu**2)**(1/3) * D_meters
    SHcr = (0.3/(1+1.2*Dstar)) + 0.055*(1-np.exp(-0.02 * Dstar))
    taucrit = rhow * (s - 1) * g * D_meters * SHcr #in Pascals
    return taucrit

def calculate_receptor_change_percentage(receptor_filename, data_diff, ofpath):
    #gdal version
    data_diff = np.transpose(data_diff)
    data = gdal.Open(receptor_filename)
    img = data.GetRasterBand(1)
    receptor_array = img.ReadAsArray()
    # transpose to be in same orientation as NetCDF
    receptor_array[receptor_array < 0] = np.nan

    pct_mobility = {'Receptor_Value': [],
                        'Increased Deposition': [],
                        'Reduced Deposition': [],
                        'Reduced Erosion': [],
                        'Increased Erosion': [],
                        'No Change':[]}
    
    for unique_val in [i for i in np.unique(receptor_array) if ~np.isnan(i)]:
        mask = receptor_array==unique_val
        data_at_val = np.where(mask, data_diff, np.nan)
        data_at_val = data_at_val.flatten()
        data_at_val = data_at_val[~np.isnan(data_at_val)]
        ncells = data_at_val.size
        pct_mobility['Receptor_Value'].append(unique_val)
        pct_mobility['Increased Deposition'].append(100 * np.size(np.flatnonzero(data_at_val==-2))/ncells)
        pct_mobility['Reduced Deposition'].append(100 * np.size(np.flatnonzero(data_at_val>-1))/ncells)
        pct_mobility['Reduced Erosion'].append(100 * np.size(np.flatnonzero(data_at_val==1))/ncells)
        pct_mobility['Increased Erosion'].append(100 * np.size(np.flatnonzero(data_at_val==2))/ncells)
        pct_mobility['No Change'].append(100 * np.size(np.flatnonzero(data_at_val==0))/ncells)

        # print(f" Receptor Value = {unique_val}um | decrease = {pct_decrease}% | increase = {pct_increase}% | no change = {pct_nochange}%")
    DF = pd.DataFrame(pct_mobility)
    DF = DF.set_index('Receptor_Value')
    DF.to_csv(os.path.join(ofpath, 'receptor_percent_change.csv'))

def calculate_taumax_structured(
    dev_present_file,
    dev_notpresent_file,
    bc_file,
    run_order_file,
    receptor_filename=None,
):
    # ===========================
    # Load With NetCDF files
    # the netcdf files here are 4 (case number, time, x, y) or 5-dimensional (case number, depth, time, x, y).
    # nc_file = glob.glob(os.path.join(run_dir,'run_dir_wecs','*.nc'))

    # Read The device present NetCDF file and parse contents needed for plotting
    file_dev_present = Dataset(dev_present_file)
    # X-coordinate of cell center
    xcor = file_dev_present.variables["XCOR"][:].data
    # Y-coordinate of cell center
    ycor = file_dev_present.variables["YCOR"][:].data
    # TAUMAX
    data_dev = file_dev_present.variables['TAUMAX'][:]
    # close the device prsent file
    file_dev_present.close()

    file_dev_notpresent = Dataset(dev_notpresent_file)
    data_nodev = file_dev_notpresent.variables['TAUMAX'][:]
    # close the device not present file
    file_dev_notpresent.close()


    # Load and parse run order file. This csv file has the wave conditions for each case. The wave conditions are listed in the order of cases as they are
    # stored in the first dimension of data_wecs or data_nodev
    df_run_order = pd.read_csv(run_order_file)

    # filter out bad runs from wecs
    if "bad_run" in df_run_order.columns:
        df_run_order = df_run_order.loc[df_run_order["bad_run"] != "X", :]

    # Load BC file with probabilities and find appropriate probability
    BC_probability = np.loadtxt(bc_file, delimiter=",", skiprows=1)

    df = pd.DataFrame(
        {
            "Hs": BC_probability[:, 0].astype(float),
            "Tp": BC_probability[:, 1].astype(int).astype(str),
            "Dir": BC_probability[:, 2].astype(int).astype(str),
            "prob": BC_probability[:, 4].astype(float) / 100.0,
        },
    )
    # generate a primary key (string) for merge with the run order based on forcing
    df["pk"] = (
        ["Hs"]
        + df["Hs"].map("{:.2f}".format)
        + ["Tp"]
        + df["Tp"].str.pad(2, fillchar="0")
        + ["Dir"]
        + df["Dir"]
    )

    # merge to the run order. This trims out runs that we want dropped.
    # set up a dataframe of probabilities    
    df_merge = pd.merge(df_run_order, df, how="left", left_on="bc_name", right_on="pk")

    # Loop through all boundary conditions and create images
    # Calculate the critical shear stress from the grain size receptor.
    if ((receptor_filename is not None) or (not receptor_filename == "")):
        data = gdal.Open(receptor_filename)
        img = data.GetRasterBand(1)
        receptor_array = img.ReadAsArray()
        # transpose to be in same orientation as NetCDF
        # receptor_array = np.transpose(receptor_array)
        receptor_array[receptor_array < 0] = 0
        # Ensure the receptor array is the same shape as the model grid.
        (upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = data.GetGeoTransform()
        cols = data.RasterXSize
        rows = data.RasterYSize
        r_rows = np.arange(rows) * y_size + upper_left_y + (y_size / 2)
        r_cols = np.arange(cols) * x_size + upper_left_x + (x_size / 2)
        r_cols = np.where(r_cols<0, r_cols+360, r_cols)
        x_grid, y_grid = np.meshgrid(r_cols, r_rows)
        receptor_array = griddata((x_grid.flatten(), y_grid.flatten()), receptor_array.flatten(), (xcor,ycor), method='nearest', fill_value=0)

        taucrit = critical_shear_stress(D_meters=receptor_array * 1e-6,
                            rhow=1024,
                            nu=1e-6,
                            s=2.65,
                            g=9.81) #units N/m2 = Pa
    else:
        # taucrit without a receptor
        #Assume the following grain sizes and conditions for typical beach sand (Nielsen, 1992 p.108)
        taucrit = critical_shear_stress(D_meters=200*1e-6, rhow=1024, nu=1e-6, s=2.65, g=9.81)  #units N/m2 = Pa


    data_dev_max = np.amax(data_dev, axis=1, keepdims=True) #look at maximum shear stress difference change
    taumax_combined_nodev = np.zeros(np.shape(data_nodev[0, 0, :, :]))
    taumax_combined_dev = np.zeros(np.shape(data_dev[0, 0, :, :]))

    for run_dev, run_nodev, prob in zip(
        df_merge["wec_run_id"],
        df_merge["nowec_run_id"],
        df_merge["prob"],
    ):
        bc_nodev_num = int(run_nodev - 1)
        bc_dev_num = int(run_dev - 1)

        

        # get last axis value
            
        taumax_combined_nodev = taumax_combined_nodev + prob * data_nodev[bc_nodev_num,-1,:,:] #tau_max #from last model run
        taumax_combined_dev = taumax_combined_dev + prob * data_dev_max[bc_dev_num,-1,:,:] #tau_max #from maximum of timeseries

        # ===============================================================
        # Compute normalized difference between with WEC and without WEC
        # QA dataframes are here
        # if np.isnan(data_wecs[run_wec, -1, :, :].data[:]).all() == True | np.isnan(data_nodev[run_nowec, -1, :, :].data[:]).all() == True:
        #    continue
        # wec_diff = wec_diff + prob*(data_w_wecs[bcnum,1,:,:] - data_wo_wecs[bcnum,1,:,:])/data_wo_wecs[bcnum,1,:,:]

    mobility_parameter_nodev = taumax_combined_nodev / taucrit
    mobility_parameter_dev = taumax_combined_dev / taucrit
    # Calculate risk metrics over all runs

    mobility_parameter_diff = mobility_parameter_dev - mobility_parameter_nodev
    tau_diff = taumax_combined_dev - taumax_combined_nodev
    mobility_classification = np.zeros(mobility_parameter_diff.shape)
    #Reduced Erosion (Tw<Tb) & (Tw-Tb)>1
    mobility_classification = np.where(((mobility_parameter_dev < mobility_parameter_nodev) & (mobility_parameter_nodev>=1)), 1, mobility_classification)
    #Increased Erosion (Tw>Tb) & (Tw-Tb)>1
    mobility_classification = np.where(((mobility_parameter_dev > mobility_parameter_nodev) & (mobility_parameter_nodev>=1)), 2, mobility_classification)
    #Reduced Deposition (Tw>Tb) & (Tw-Tb)<1
    mobility_classification = np.where(((mobility_parameter_dev > mobility_parameter_nodev) & (mobility_parameter_nodev<1)), -1, mobility_classification)
    #Increased Deposition (Tw>Tb) & (Tw-Tb)>1
    mobility_classification = np.where(((mobility_parameter_dev < mobility_parameter_nodev) & (mobility_parameter_nodev<1)), -2, mobility_classification)
    #NoChange = 0

    # ========================================================

    # convert to a geotiff, using wec_diff

    listOfFiles = [mobility_parameter_nodev, mobility_parameter_dev, mobility_parameter_diff, tau_diff, mobility_classification]

    # return the number of listOfFiles
    return listOfFiles


def calculate_diff_cec(folder_base, folder_cec, taucrit=100.0):

    """
    Given non linear grid files calculate the difference.

    Currently taucrit is 100. Would need to make
    constant raster of from either fname_base or fname_cec for raster use.
    """
    # Loop through the base folder name and the cec folders, Get the return interval from the filename
    first_run = True
    for fname_base, fname_cec in zip(
        glob.glob(os.path.join(folder_base, "*.nc")),
        glob.glob(os.path.join(folder_cec, "*.nc")),
    ):

        # get the return interval from the name
        return_interval = int(
            os.path.basename(fname_base).split("_")[1].split("tanana")[1],
        )

        f_base = Dataset(fname_base, mode="r", format="NETCDF4")
        f_cec = Dataset(fname_cec, mode="r", format="NETCDF4")

        tau_base = f_base.variables["taus"][1, :].data
        tau_cec = f_cec.variables["taus"][1, :].data

        lon = f_base.variables["FlowElem_xcc"][:].data
        lat = f_base.variables["FlowElem_ycc"][:].data

        if first_run:
            cec_diff_bs = np.zeros(np.shape(tau_base))
            cec_diff_cecs = np.zeros(np.shape(tau_base))
            cec_diff = np.zeros(np.shape(tau_base))
            first_run = False

        # lon, lat = transformer.transform(f_base.variables['NetNode_x'][:].data, f_base.variables['NetNode_y'][:].data)
        # df = pd.DataFrame({'lon': lon, 'lat':lat})
        # df.to_csv('out_test_lon_lat.csv', index = False)

        # taucrit = 1.65*980*((1.9e-4*10**6)/10000)*0.0419
        taucrit = taucrit
        # return_interval = 1
        prob = 1 / return_interval

        # calculate differences
        cec_diff_bs = cec_diff_bs + prob * tau_base / (taucrit * 10)
        cec_diff_cecs = cec_diff_cecs + prob * tau_cec / (taucrit * 10)

        cec_diff_df = cec_diff_cecs - cec_diff_bs

        # transpose and flip
        newarray = np.transpose(cec_diff_df)
        array2 = np.flip(newarray, axis=0)

        # cec_diff_df = pd.DataFrame(array2)
        # cec_diff_df.to_csv(fr'C:\Users\ependleton52\Documents\Projects\Sandia\SEAT_plugin\Code_Model\Codebase\tanana\out_cec_{int(return_interval)}.csv', index = False)

    # adjust the signs. Take from Kaus' oroginal code
    cec_diff_bs_sgn = np.floor(cec_diff_bs * 25) / 25
    cec_diff_cecs_sgn = np.floor(cec_diff_cecs * 25) / 25
    cec_diff = np.sign(cec_diff_cecs_sgn - cec_diff_bs_sgn) * cec_diff_cecs_sgn
    cec_diff = cec_diff.astype(int) + cec_diff_cecs - cec_diff_bs
    # cec_diff[np.abs(cec_diff)<0.001] = 0

    # Use triangular interpolation to generate grid. Matched x, y counts
    # reflon=np.linspace(lon.min(),lon.max(),1000)
    # reflat=np.linspace(lat.min(),lat.max(),1000)
    reflon = np.linspace(lon.min(), lon.max(), 169)
    reflat = np.linspace(lat.min(), lat.max(), 74)

    # create long, lat from the meshgrid
    reflon, reflat = np.meshgrid(reflon, reflat)

    # original
    flatness = 0.1  # flatness is from 0-.5 .5 is equilateral triangle
    flatness = 0.2  # flatness is from 0-.5 .5 is equilateral triangle
    tri = Triangulation(lon, lat)
    mask = TriAnalyzer(tri).get_flat_tri_mask(flatness)
    tri.set_mask(mask)
    tli = LinearTriInterpolator(tri, cec_diff)
    tau_interp = tli(reflon, reflat)

    newarray = np.transpose(tau_interp[:, :].data)
    array2 = np.flip(tau_interp[:, :].data, axis=0)

    # rows, cols = np.shape(tau_interp)
    rows, cols = np.shape(array2)

    return (rows, cols, array2)


def create_raster(
    output_path,
    cols,
    rows,
    nbands,
):

    """Create a gdal raster object."""
    # create gdal driver - doing this explicitly
    driver = gdal.GetDriverByName(str("GTiff"))

    output_raster = driver.Create(
        output_path,
        int(cols),
        int(rows),
        nbands,
        eType=gdal.GDT_Float32,
    )

    # spatial_reference = osr.SpatialReference()
    # spatial_reference.ImportFromEPSG(spatial_reference_system_wkid)
    # output_raster.SetProjection(spatial_reference.ExportToWkt())

    # returns gdal data source raster object
    return output_raster


def numpy_array_to_raster(
    output_raster,
    numpy_array,
    bounds,
    cell_resolution,
    spatial_reference_system_wkid,
    output_path,
):

    """Create the output raster."""
    # create output raster
    # (upper_left_x, x_resolution, x_skew 0, upper_left_y, y_skew 0, y_resolution).
    # Need to rotate to go from np array to geo tiff. This can vary depending on the methods used above. Will need to test for this.
    geotransform = (
        bounds[0],
        cell_resolution[0],
        0,
        bounds[1] + cell_resolution[1],
        0,
        -1 * cell_resolution[1],
    )

    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromEPSG(spatial_reference_system_wkid)

    output_raster.SetProjection(
        spatial_reference.ExportToWkt(),
    )  # exports the cords to the file
    output_raster.SetGeoTransform(geotransform)
    output_band = output_raster.GetRasterBand(1)
    # output_band.SetNoDataValue(no_data) #Not an issue, may be in other cases?
    output_band.WriteArray(numpy_array)

    output_band.FlushCache()
    output_band.ComputeStatistics(
        False,
    )  # you want this false, true will make computed results, but is faster, could be a setting in the UI perhaps, esp for large rasters?

    if os.path.exists(output_path) == False:
        raise Exception("Failed to create raster: %s" % output_path)

    # this closes the file
    output_raster = None
    return output_path


# now call the functions
if __name__ == "__main__":

    """Testing paramters."""

    # =================
    # User input block

    # Set directory with output folders (contains with_wecs and without_wecs folders)
    # run_dir = r'C:\Users\mjamieson61\Documents\Internal_Working\Projects\QGIS_Python\Codebase'
    # linux

    # Set plot variable
    # plotvar = 'VEL'      # Concentrations per layer at zeta point
    plotvar = "TAUMAX"  # Tau max in zeta points (N/m^2)
    # plotvar = 'DPS'     # Bottom depth at zeta point (m)

    # Set NetCDF file to load WEC
    dev_present_file = r"H:\Projects\C1308_SEAT\SEAT_inputs\plugin-input\oregon\devices-present\trim_sets_flow_inset_allruns.nc"
    dev_notpresent_file = r"H:\Projects\C1308_SEAT\SEAT_inputs\plugin-input\oregon\devices-not-present\trim_sets_flow_inset_allruns.nc"

    # cec files
    # dev_present_file = r"C:\Users\ependleton52\Desktop\temp_local\QGIS\Code_Model\Codebase\cec\with_cec_1.nc"
    # dev_notpresent_file = r"C:\Users\ependleton52\Desktop\temp_local\QGIS\Code_Model\Codebase\cec\no_cec_1.nc"

    # dev_present_file = r"C:\Users\ependleton52\Desktop\temp_local\QGIS\Code_Model\Codebase\tanana\DFM_OUTPUT_tanana100\modified\tanana100_map_0_tanana1_cec.nc"
    # dev_notpresent_file = r"C:\Users\ependleton52\Desktop\temp_local\QGIS\Code_Model\Codebase\tanana\DFM_OUTPUT_tanana100\modified\tanana100_map_6_tanana1_cec.nc"

    # set the boundary_coditions file
    bc_file = r"H:\Projects\C1308_SEAT\SEAT_inputs\plugin-input\oregon\boundary-condition\BC_probability_Annual_SETS.csv"

    # run order file
    run_order_file = r"H:\Projects\C1308_SEAT\SEAT_inputs\plugin-input\oregon\run-order\run_order_wecs_bad_runs_removed_v2.csv"

    receptor = r"H:\Projects\C1308_SEAT\SEAT_inputs\plugin-input\oregon\receptor\grainsize_receptor.tif"

    # configuration for raster translate
    GDAL_DATA_TYPE = gdal.GDT_Float32
    GEOTIFF_DRIVER_NAME = r"GTiff"

    # Skip the bad runs for now
    bcarray = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 22],
    )

    # all runs
    # bcarray = [i for i in range(1,23)]

    # SWAN will always be in meters. Not always WGS84
    SPATIAL_REFERENCE_SYSTEM_WKID = 4326  # WGS84 meters
    nbands = 1  # should be one always right?
    # bounds = [-124.2843933,44.6705] #x,y or lon,lat, this is pulled from an input data source
    # cell_resolution = [0.0008,0.001 ] #x res, y res or lon, lat, same as above

    # from Kaus -235.8+360 degrees = 124.2 degrees. The 235.8 degree conventions follows longitudes that increase
    # eastward from Greenwich around the globe. The 124.2W, or -124.2 goes from 0 to 180 degrees to the east of Greenwich.
    bounds = [
        xcor.min() - 360,
        ycor.min(),
    ]  # x,y or lon,lat, this is pulled from an input data source
    # look for dx/dy
    dx = xcor[1, 0] - xcor[0, 0]
    dy = ycor[0, 1] - ycor[0, 0]
    cell_resolution = [dx, dy]

    # will we ever need to do a np.isnan test?
    # NO_DATA = 'NaN'

    # set output path
    output_path = r"C:\Users\ependleton52\Desktop\temp_local\QGIS\Code_Model\Codebase\rasters\rasters_created\plugin\out_calculated.tif"

    # Functions
    rows, cols, numpy_array = transform_netcdf(
        dev_present_file,
        dev_notpresent_file,
        bc_file,
        run_order_file,
        bcarray,
        plotvar,
    )

    output_raster = create_raster(
        output_path,
        cols,
        rows,
        nbands,
    )

    # post processing of numpy array to output raster
    output_raster = numpy_array_to_raster(
        output_raster,
        numpy_array,
        bounds,
        cell_resolution,
        SPATIAL_REFERENCE_SYSTEM_WKID,
        output_path,
    )

    """
    # add the raster to the QGIS interface
    #newRLayer = iface.addRasterLayer(output_raster)

    #now add to the interface of QGIS
    #newRLayer = iface.addRasterLayer(output_path)
    """
