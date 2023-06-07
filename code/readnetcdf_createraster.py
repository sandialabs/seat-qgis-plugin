#!/usr/bin/python

# Oregon Coast Delft3D WEC Difference Plotting

# Plot normalized comparison of simulations with WECs and without WECS for
# user selected variable for all boundary conditions

# Usage:
# python delft_wecs_diff_all_bcs.py

# Output: # of BCs figures saved to run_dir directory

# Revision Log
# -----------------------------------------------------------------------------
# 09/12/2017 Chris Flanary - created script
# 09/18/2017 CF - added file_num variable for multiple NetCDF files and CSV
#                 file read for run order
# 12/20/2017 Kaus added risk assessment code
# 12/22/2017 Kaus added creation of metric file for habitat polygons
# 06/09/2020 Kaus simplified script for a single run, for Matt
# 06/26/2020 Matt added GDAL geotransform
# 05/10/2022 Eben Pendleton added comments

import glob
import os

import numpy as np
import pandas as pd
from matplotlib.tri import LinearTriInterpolator, TriAnalyzer, Triangulation
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
    taucrit = (2.65 - 1) * g * D_meters * SHcr #in Pascals
    return taucrit

def calculate_receptor_change_percentage(receptor_filename, data_diff, ofpath):
    #gdal version
    data_diff = np.transpose(data_diff)
    data = gdal.Open(receptor_filename)
    img = data.GetRasterBand(1)
    receptor_array = img.ReadAsArray()
    # transpose to be in same orientation as NetCDF
    receptor_array = np.transpose(receptor_array)
    receptor_array[receptor_array < 0] = 0

    pct_mobility = {'Receptor_Value': [],
                        'pct_decrease': [],
                        'pct_increase': [],
                        'pct_nochange': []}
    for unique_val in np.unique(receptor_array):
        mask = receptor_array==unique_val
        data_at_val = np.where(mask, data_diff, np.nan)
        data_at_val = data_at_val.flatten()
        data_at_val = data_at_val[~np.isnan(data_at_val)]
        ncells = data_at_val.size
        pct_mobility['Receptor_Value'].append(unique_val)
        pct_mobility['pct_decrease'].append(100 * np.size(np.flatnonzero(data_at_val<0))/ncells)
        pct_mobility['pct_increase'].append(100 * np.size(np.flatnonzero(data_at_val>0))/ncells)
        pct_mobility['pct_nochange'].append(100 * np.size(np.flatnonzero(data_at_val==0))/ncells)

        # print(f" Receptor Value = {unique_val}um | decrease = {pct_decrease}% | increase = {pct_increase}% | no change = {pct_nochange}%")
    DF = pd.DataFrame(pct_mobility)
    DF = DF.set_index('Receptor_Value')
    DF.to_csv(os.path.join(ofpath, 'receptor_percent_change.csv'))

def transform_netcdf_ro(
    dev_present_file,
    dev_notpresent_file,
    bc_file,
    run_order_file,
    plotvar,
    receptor_filename=None,
    receptor=True,
):
    # ===========================
    # Load With WECs NetCDF file.
    # MATT: the netcdf files here are 4 (case number, time, x, y) or 5-dimensional (case number, depth, time, x, y).
    # We first load the netcdf files when wecs are present, into variable data_wecs, then (below) load netcdf files without wecs into data_bs
    # Find name of NetCDF output file
    # nc_file = glob.glob(os.path.join(run_dir,'run_dir_wecs','*.nc'))

    # Read The device present NetCDF file and parse contents needed for plotting
    file_dev_present = Dataset(dev_present_file)

    # X-coordinate of cell center
    xcor = file_dev_present.variables["XCOR"][:].data
    # Y-coordinate of cell center
    ycor = file_dev_present.variables["YCOR"][:].data

    # if we are running a structured case update the plot varaiable to what we can query
    if plotvar == "TAUMAX -Structured":
        plotvar = "TAUMAX"

    # Deprecated
    # delft_time = file.variables['time'][:]
    # depth = file.variables['DPS0'][:] # Initial bottom depth at zeta points (positive down)
    # sed_fracs = np.squeeze(file.variables['LYRFRAC'][0,0,:,0,:,:])
    if plotvar != "VEL": #not velocity
        # set as 4D netcdf files
        data_wecs = file_dev_present.variables[plotvar][:]

    else: #Larval Transport Velocity
        # 5D netcdf files. Pick last depth that corresponds to bottom
        u = file_dev_present.variables["U1"][:, :, -1, :, :]
        v = file_dev_present.variables["V1"][:, :, -1, :, :]
        data_wecs = np.sqrt(u**2 + v**2)

    # close the device prsent file
    file_dev_present.close()

    if plotvar == "TAUMAX": # Grain Size
        # Empirical calculation of Sediment D50s and critical shear stress for erosion
        # nsed = np.shape(sed_fracs)[0]
        # sed_d50 = np.array([3.5e-4, 2.75e-4, 2.0e-4, 0.75e-4])
        # layer_d50 = np.zeros((np.shape(sed_fracs)[1], np.shape(sed_fracs)[2]))
        # for ised in range(0,nsed):
        #    layer_d50 = layer_d50 + np.squeeze(sed_fracs[ised,:,:])*sed_d50[ised]
        #
        # taucrit = 1.65*980*((layer_d50*10**6)/10000)*0.0419

        # E & E case
        # soil density * gravity * grain size(m) * 10^6 / unit converter *
        # taucrit = 1.65*980*((1.9e-4*10**6)/10000)*0.0419

        # Add in the receptor here. This started as grain size but is geeralized
        if receptor_filename is not None:
            data = gdal.Open(receptor_filename)
            img = data.GetRasterBand(1)
            receptor_array = img.ReadAsArray()
            # transpose to be in same orientation as NetCDF
            receptor_array = np.transpose(receptor_array)
            receptor_array[receptor_array < 0] = 0
            # soil density * gravity * grain size array * 10^6 / unit converter *
            # taucrit = 1.65*980*((gs_array*10**6)/10000)*0.0419

            # convert micron to cm 1 micron is 1e-4
            # soil density (g/cm^2) * standard gravity (cm/s^2) * (micron / (convert to cm)) * unit conversion
            # taucrit = 1.65 * 980 * ((receptor_array) / 10000) * 0.0419
            taucrit = critical_shear_stress(D_meters=receptor_array * 1e-6,
                                rhow=1024,
                                nu=1e-6,
                                s=2.65,
                                g=9.81) #units N/m2 = Pa
        else:
            # taucrit without a receptor
            # taucrit = 1.65 * 980 * 0.0419
            #Assume the following grain sizes and conditions for typical beach sand (Nielsen, 1992 p.108)
            taucrit = critical_shear_stress(D_meters=200*1e-6, rhow=1024, nu=1e-6, s=2.65, g=9.81) #units N/m2 = Pa

    elif plotvar == "VEL": #Larval Transport
        # Define critical velocity for motility as 0.05 m/s
        velcrit = 0.05 * np.ones(np.shape(np.squeeze(u[0, 0, :, :])))

    # Load and parse run order file. This csv file has the wave conditions for each case. The wave conditions are listed in the order of cases as they are
    # stored in the first dimension of data_wecs or data_nodev
    df_run_order = pd.read_csv(run_order_file)

    # filter out bad runs from wecs
    df_run_order = df_run_order.loc[df_run_order["bad_run"] != "X", :]

    # Load BC file with probabilities and find appropriate probability
    BC_Annie = np.loadtxt(bc_file, delimiter=",", skiprows=1)

    # ==============================
    # Load WECs NetCDF file without wecs into variable data_nodev

    # Find name of NetCDF device not present output file
    # Read NetCDF file and parse contents needed for plotting
    file = Dataset(dev_notpresent_file)
    if plotvar != "VEL":
        data_nodev = file.variables[plotvar][:]
    else:
        u = file.variables["U1"][:, :, -1, :, :]
        v = file.variables["V1"][:, :, -1, :, :]
        #data_bs
        data_nodev = np.sqrt(u**2 + v**2)

    # close the device not present file
    file.close()

    # create zero arrays for the device present / not present
    wec_diff_nodev = np.zeros(np.shape(data_nodev[0, 0, :, :]))
    wec_diff_wecs = np.zeros(np.shape(data_wecs[0, 0, :, :]))
    # wec_diff = np.zeros(np.shape(data_wecs[0,0,:,:]))

    # flip the data arrays
    data_nodev = np.flip(data_nodev, axis=3)
    data_wecs = np.flip(data_wecs, axis=3)

    # =======================================================
    # set up a dataframe of probabilities
    df = pd.DataFrame(
        {
            "Hs": BC_Annie[:, 0].astype(float),
            "Tp": BC_Annie[:, 1].astype(int).astype(str),
            "Dir": BC_Annie[:, 2].astype(int).astype(str),
            "prob": BC_Annie[:, 4].astype(float) / 100.0,
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
    # df_merge = pd.merge(df_run_order, df_ro_no_wecs, how = 'left', on = 'bc_name')
    df_merge = pd.merge(df_run_order, df, how="left", left_on="bc_name", right_on="pk")
    # Loop through all boundary conditions and create images
    # breakpoint()
    for run_wec, run_nowec, prob in zip(
        df_merge["wec_run_id"],
        df_merge["nowec_run_id"],
        df_merge["prob"],
    ):
        bcnum = int(run_nowec - 1)

        # ===============================================================
        # Compute normalized difference between with WEC and without WEC
        # QA dataframes are here
        # if np.isnan(data_wecs[run_wec, -1, :, :].data[:]).all() == True | np.isnan(data_nodev[run_nowec, -1, :, :].data[:]).all() == True:
        #    continue
        # wec_diff = wec_diff + prob*(data_w_wecs[bcnum,1,:,:] - data_wo_wecs[bcnum,1,:,:])/data_wo_wecs[bcnum,1,:,:]

        if plotvar == "TAUMAX":

            # if the shapes are the same then process. Otherwise, process to an array and stop
            if data_nodev[bcnum, -1, :, :].shape == taucrit.shape:

                # get max along the 2nd axis (time)
                data_wecs_max = np.amax(data_wecs, axis=1, keepdims=True)

                # get last axis value
                # data_wecs_max = data_wecs[:,[-1],:,:]

                # make a backup just in case
                # wec_diff_bs_b = wec_diff_bs
                # wec_diff_wecs_b = wec_diff_wecs

                if receptor == True:
                    wec_diff_nodev = wec_diff_nodev + prob * data_nodev[bcnum,-1,:,:] / taucrit
                    wec_diff_wecs = wec_diff_wecs + prob * data_wecs_max[bcnum,-1,:,:] / taucrit
                else:
                    wec_diff_nodev = (wec_diff_nodev + prob * data_nodev[bcnum, -1, :, :])
                    wec_diff_wecs = (wec_diff_wecs + prob * data_wecs_max[bcnum, -1, :, :])
                # pd.DataFrame(wec_diff_nodev).to_csv(r"H:\Projects\C1308_SEAT\SEAT_outputs\tester.csv")
                # with open("H:\Projects\C1308_SEAT\SEAT_outputs\tester_size", 'w') as file:
                #     file.write(f'{np.shape(wec_diff_nodev)}')
                # # create dataframe of subtraction for QA
                # wec_diff_df = (
                #     wec_diff_bs + prob * data_nodev[bcnum, -1, :, :]
                # ) - (wec_diff_wecs + prob * data_wecs[bcnum -1, :, :])
                # wec_diff_df = wec_diff_bs + prob * data_nodev[bcnum, -1, :, :]
                # wec_diff_df = np.transpose(wec_diff_df)
                # # removed flip 05/27/2022
                # # wec_diff_df = np.flip(wec_diff_df, axis=0)
                # wec_diff_df = pd.DataFrame(wec_diff_df)

                # dump for QA. Should make this more flexible
                # wec_diff_df.to_csv(fr'C:\Users\ependleton52\Documents\Projects\Sandia\SEAT_plugin\Code_Model\Codebase\oregon_coast_models\dataframes\out_wec{int(run_wec)}_nowec{int(run_nowec)}.csv', index = False)
                # breakpoint()
            else:
                newarray = np.transpose(data_nodev[bcnum, -1, :, :].data)
                array2 = np.flip(newarray, axis=0)
                numpy_array = array2
                rows, cols = numpy_array.shape
                breakpoint()
                # will need to dump to raster to check
                # will error as output path is not defined.
                # SPATIAL_REFERENCE_SYSTEM_WKID = 4326 #WGS84 meters
                # nbands = 1
                # bottom left, x, y netcdf file

                # from Kaus -235.8+360 degrees = 124.2 degrees. The 235.8 degree conventions follows longitudes that increase
                # eastward from Greenwich around the globe. The 124.2W, or -124.2 goes from 0 to 180 degrees to the east of Greenwich.
                bounds = [
                    xcor.min() - 360,
                    ycor.min(),
                ]  # x,y or lon,lat, this is pulled from an input data source
                # look for dx/dy
                # dx = xcor[1,0] - xcor[0,0]
                # dy = ycor[0,1] - ycor[0,0]
                # cell_resolution = [dx,dy ] #x res, y res or lon, lat, same as above

                # output_raster = create_raster(output_path,
                #      cols,
                #      rows,
                #      nbands)

                # output_raster = numpy_array_to_raster(output_raster,
                #              numpy_array,
                #              bounds,
                #              cell_resolution,
                #             SPATIAL_REFERENCE_SYSTEM_WKID, output_path)

        elif plotvar == "VEL":
            # breakpoint()
            # wec_diff_nodev = wec_diff_nodev + prob*(2*data_bs[bcnum,1,:,:] - data_bs[bcnum,1,:,:])/(velcrit*10)
            # wec_diff_wecs = wec_diff_wecs + prob*(2*data_wecs[bcnum,1,:,:] - data_wecs[bcnum,1,:,:])/(velcrit*10)
            # Should this be 0?
            wec_diff_nodev = wec_diff_nodev + prob * (
                2 * data_nodev[bcnum, 0, :, :] - data_nodev[bcnum, 0, :, :]
            ) / (velcrit * 10)
            wec_diff_wecs = wec_diff_wecs + prob * (
                2 * data_wecs[bcnum, 0, :, :] - data_wecs[bcnum, 0, :, :]
            ) / (velcrit * 10)

        elif plotvar == "DPS":
            wec_diff_nodev = wec_diff_nodev + prob * data_nodev[bcnum, 1, :, :]
            wec_diff_wecs = wec_diff_wecs + prob * data_wecs[bcnum, 1, :, :]
            # wec_diff = (data_w_wecs[bcnum,1,:,:] - data_wo_wecs[bcnum,1,:,:])/data_wo_wecs[bcnum,1,:,:]
    # ========================================================

    # Calculate risk metrics over all runs
    if plotvar == "TAUMAX" or plotvar == "VEL":
        wec_diff_nodev_sgn = np.floor(wec_diff_nodev * 25) / 25
        wec_diff_wecs_sgn = np.floor(wec_diff_wecs * 25) / 25

        wec_diff = np.sign(wec_diff_wecs_sgn - wec_diff_nodev_sgn) * wec_diff_wecs_sgn
        wec_diff = wec_diff.astype(int) + wec_diff_wecs - wec_diff_nodev

        # set to zero. Might be turning this back on
        # wec_diff[np.abs(wec_diff)<0.01] = 0

    elif plotvar == "DPS":
        wec_diff = wec_diff_wecs - wec_diff_nodev
        wec_diff[np.abs(wec_diff) < 0.0005] = 0

    # ========================================================

    # convert to a geotiff, using wec_diff

    # listOfFiles = [wec_diff_nodev, wec_diff_wecs, wec_diff, wec_diff_nodev_sgn, wec_diff_wecs_sgn]

    # transpose and pull
    newarray = np.transpose(wec_diff)
    # we should not flip here
    # array2 = np.flip(newarray, axis=0)
    array2 = newarray
    rows, cols = array2.shape

    # return the number of rows and cols and array2
    return (rows, cols, array2)


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
    bc_file = r"H:\Projects\C1308_SEAT\SEAT_inputs\plugin-input\oregon\boundary-condition\BC_Annie_Annual_SETS.csv"

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
