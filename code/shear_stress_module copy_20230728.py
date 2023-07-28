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

def estimate_grid_spacing(x,y, nsamples=100):
    import random
    import sys
    coords = list(set(zip(x,y)))
    if nsamples != len(x):
        points = [random.choice(coords) for i in range(nsamples)] # pick N random points
    else:
        points = coords
    MD = []
    for p0x, p0y in points:
        minimum_distance = sys.maxsize
        for px, py in coords:
            distance = np.sqrt((p0x - px) ** 2 + (p0y - py) ** 2)
            if (distance < minimum_distance) & (distance !=0):
                minimum_distance = distance
        MD.append(minimum_distance)
    dxdy = np.median(MD)
    return dxdy


def calc_receptor_taucrit(receptor_filename, x, y, latlon=False):
    if ((receptor_filename is not None) or (not receptor_filename == "")):
        data = gdal.Open(receptor_filename)
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
        receptor_array = griddata((x_grid.flatten(), y_grid.flatten()), receptor_array.flatten(), (x,y), method='nearest', fill_value=0)

        taucrit = critical_shear_stress(D_meters=receptor_array * 1e-6,
                            rhow=1024,
                            nu=1e-6,
                            s=2.65,
                            g=9.81) #units N/m2 = Pa
    else:
        # taucrit without a receptor
        #Assume the following grain sizes and conditions for typical beach sand (Nielsen, 1992 p.108)
        taucrit = critical_shear_stress(D_meters=200*1e-6, rhow=1024, nu=1e-6, s=2.65, g=9.81)  #units N/m2 = Pa
    return taucrit, receptor_array

def create_structured_array_from_unstructured(x, y, z, dxdy, flatness=0.2):
    # flatness is from 0-.5 .5 is equilateral triangle
    refx = np.arange(np.nanmin(x), np.nanmax(x)+dxdy, dxdy)
    refy = np.arange(np.nanmin(y), np.nanmax(y)+dxdy, dxdy)
    refxg, refyg = np.meshgrid(refx, refy)
    tri = Triangulation(x, y)
    mask = TriAnalyzer(tri).get_flat_tri_mask(flatness)
    tri.set_mask(mask)
    tli = LinearTriInterpolator(tri, z)
    z_interp = tli(refxg, refyg)
    return refxg, refyg, z_interp.data

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
           
        taumax_combined_nodev = taumax_combined_nodev + prob * data_nodev[bc_nodev_num,-1,:,:] #tau_max #from last model run
        taumax_combined_dev = taumax_combined_dev + prob * data_dev_max[bc_dev_num,-1,:,:] #tau_max #from maximum of timeseries

        # ===============================================================
        # Compute difference between with WEC and without WEC
    
    tau_diff = taumax_combined_dev - taumax_combined_nodev

    taucrit, receptor_array = calc_receptor_taucrit(receptor_filename, xcor, ycor, latlon=True)
    mobility_parameter_nodev = taumax_combined_nodev / taucrit
    mobility_parameter_nodev = np.where(receptor_array==0, 0, mobility_parameter_nodev)
    mobility_parameter_dev = taumax_combined_dev / taucrit
    mobility_parameter_dev = np.where(receptor_array==0, 0, mobility_parameter_dev)
    # Calculate risk metrics over all runs

    mobility_parameter_diff = mobility_parameter_dev - mobility_parameter_nodev

    mobility_classification = classify_mobility(mobility_parameter_dev, mobility_parameter_nodev)

    listOfFiles = [tau_diff, mobility_parameter_nodev, mobility_parameter_dev, mobility_parameter_diff, mobility_classification]

    # return the number of listOfFiles
    dx = np.nanmean(np.diff(xcor[:,0]))
    dy = np.nanmean(np.diff(ycor[0,:]))
    return listOfFiles, xcor, ycor, dx, dy


def calculate_taumax_unstructured(fpath_nodev, fpath_dev, receptor_filename):
    """
    Given unstructured grid files calculate the difference.
    """
    # glob.glob(os.path.join(fpath_nodev, "*.nc")
    files_nodev = [i for i in os.listdir(fpath_nodev) if i.endswith('.nc')]
    files_dev = [i for i in os.listdir(fpath_dev) if i.endswith('.nc')]
    #extract return order from file name, file must be in format x_xxx_returninverval_xxx.nc, example: 1_tanana_1_map.nc
    return_intervals_nodev = np.zeros((len(files_nodev)))
    for ic, file in enumerate(files_nodev):
        return_intervals_nodev[ic] = float(file.split('.')[0].split('_')[2])
    return_intervals_dev = np.zeros((len(files_dev)))
    for ic, file in enumerate(files_dev):
        return_intervals_dev[ic] = float(file.split('.')[0].split('_')[2])

    #ensure return order for nodev matches dev files
    if np.any(return_intervals_nodev != return_intervals_dev):
        adjust_dev_order = []
        for ri in return_intervals_dev:
            adjust_dev_order = np.append(adjust_dev_order, np.flatnonzero(return_intervals_dev == ri))
        adjust_dev_order
        files_dev = [files_dev[int(i)] for i in adjust_dev_order]
        return_intervals_dev = [return_intervals_dev[int(i)] for i in adjust_dev_order]

    DF = pd.DataFrame({'files_nodev':files_nodev, 
                    'return_intervals_nodev':return_intervals_nodev,
                    'files_dev':files_dev,
                    'return_intervals_dev':return_intervals_dev})

    first_run = True
    for ir, row in DF.iterrows():
        # DS_nodev = xr.open_dataset(os.path.join(fpath_nodev, row.files_nodev))
        DS_nodev = Dataset(os.path.join(fpath_nodev, row.files_nodev))
        # DS_dev = xr.open_dataset(os.path.join(fpath_dev, row.files_dev))
        DS_dev = Dataset(os.path.join(fpath_dev, row.files_dev))
        tau_nodev = DS_nodev.variables["taus"][-1,:].data
        tau_dev = DS_dev.variables["taus"][-1,:].data

        if first_run:
            xc = DS_nodev.variables["FlowElem_xcc"][:].data
            yc = DS_nodev.variables["FlowElem_ycc"][:].data
            taumax_combined_nodev = 1/row.return_intervals_nodev * tau_nodev 
            taumax_combined_dev = 1/row.return_intervals_dev * tau_dev 
            first_run = False
            
        DS_dev.close()
        DS_nodev.close()
        taumax_combined_nodev = taumax_combined_nodev + 1/row.return_intervals_nodev * tau_nodev 
        taumax_combined_dev = taumax_combined_dev + 1/row.return_intervals_dev * tau_dev 

    #calculate stressor diff
    tau_diff = taumax_combined_dev - taumax_combined_nodev

    #calculate mobility diff
    taucrit, receptor_array = calc_receptor_taucrit(receptor_filename, xc, yc)
    mobility_parameter_nodev = taumax_combined_nodev / taucrit
    mobility_parameter_nodev = np.where(receptor_array==0, 0, mobility_parameter_nodev)
    mobility_parameter_dev = taumax_combined_dev / taucrit
    mobility_parameter_dev = np.where(receptor_array==0, 0, mobility_parameter_dev)
    mobility_parameter_diff = mobility_parameter_dev - mobility_parameter_nodev

    #create structured output from the unstructured input
    dxdy = estimate_grid_spacing(xc,yc, nsamples=100)
    rx, ry, tau_diff_struct = create_structured_array_from_unstructured(xc, yc, tau_diff, dxdy, flatness=0.2)
    _, _, mobility_parameter_nodev_struct = create_structured_array_from_unstructured(xc, yc, mobility_parameter_nodev, dxdy, flatness=0.2)
    _, _, mobility_parameter_dev_struct = create_structured_array_from_unstructured(xc, yc, mobility_parameter_dev, dxdy, flatness=0.2)
    _, _, mobility_parameter_diff_struct = create_structured_array_from_unstructured(xc, yc, mobility_parameter_diff, dxdy, flatness=0.2)

    mobility_classification = classify_mobility(mobility_parameter_dev_struct, mobility_parameter_nodev_struct)

    listOfFiles = [tau_diff_struct, mobility_parameter_nodev_struct, mobility_parameter_dev_struct, mobility_parameter_diff_struct, mobility_classification]
    return listOfFiles, rx, ry, dxdy, dxdy

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
