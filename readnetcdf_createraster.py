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

# Turn off display of figure (Linux command line only)
#import matplotlib
#matplotlib.use('Agg')
import glob, os, osr

import numpy as np
import pandas as pd

from netCDF4 import Dataset
import re
import gdal

from matplotlib.tri import Triangulation, TriAnalyzer, LinearTriInterpolator
# these imports currently don't work :(
# from qgis.core import *
# import qgis.utils
    
def transform_netcdf_ro(dev_present_file, dev_notpresent_file, bc_file, run_order_file, plotvar, receptor_filename = None):
    #===========================
    # Load With WECs NetCDF file.
    # MATT: the netcdf files here are 4 (case number, time, x, y) or 5-dimensional (case number, depth, time, x, y). 
    # We first load the netcdf files when wecs are present, into variable data_wecs, then (below) load netcdf files without wecs into data_bs
    # Find name of NetCDF output file
    # nc_file = glob.glob(os.path.join(run_dir,'run_dir_wecs','*.nc'))

    # Read The device present NetCDF file and parse contents needed for plotting
    file = Dataset(dev_present_file)
    # lat  = file.variables['YZ'][:] # Y-coordinate of cell center
    # lon  = file.variables['XZ'][:] # X-coordinate of cell center
    
    xcor = file.variables['XCOR'][:].data
    ycor = file.variables['YCOR'][:].data
    
    if plotvar == 'TAUMAX -Structured': plotvar = 'TAUMAX'
    
    # Deprecated?
    # delft_time = file.variables['time'][:]
    # depth = file.variables['DPS0'][:] # Initial bottom depth at zeta points (positive down)
    #sed_fracs = np.squeeze(file.variables['LYRFRAC'][0,0,:,0,:,:])
    if plotvar != 'VEL':
        # 4D netcdf files
        data_wecs = file.variables[plotvar][:]
        
    else:
        # 5D netcdf files. Pick last depth that corresponds to bottom
        u = file.variables['U1'][:,:,-1,:,:]
        v = file.variables['V1'][:,:,-1,:,:]
        data_wecs = np.sqrt(u**2 + v**2)
        
    file.close()

    if plotvar == 'TAUMAX':
        # Empirical calculation of Sediment D50s and critical shear stress for erosion
        #nsed = np.shape(sed_fracs)[0]
        #sed_d50 = np.array([3.5e-4, 2.75e-4, 2.0e-4, 0.75e-4])
        #layer_d50 = np.zeros((np.shape(sed_fracs)[1], np.shape(sed_fracs)[2]))
        #for ised in range(0,nsed):
        #    layer_d50 = layer_d50 + np.squeeze(sed_fracs[ised,:,:])*sed_d50[ised]
        #    
        #taucrit = 1.65*980*((layer_d50*10**6)/10000)*0.0419
        
        # E & E case
        # soil density * gravity * grain size(m) * 10^6 / unit converter *
        #taucrit = 1.65*980*((1.9e-4*10**6)/10000)*0.0419
        
        # ADD in receptor here. This started as grain size but is geeralized
        if receptor_filename is not None:        
            data = gdal.Open(receptor_filename)
            img = data.GetRasterBand(1)
            receptor_array = img.ReadAsArray()
            # transpose to be in same orientation as NetCDF
            receptor_array = np.transpose(receptor_array)
            receptor_array[receptor_array < 0] = 0
            # soil density * gravity * grain size array * 10^6 / unit converter *
            #taucrit = 1.65*980*((gs_array*10**6)/10000)*0.0419
            
            # convert micron to cm 1 micron is 1e-4
            # soil density (g/cm^2) * standard gravity (cm/s^2) * (micron / (convert to cm)) * unit conversion
            taucrit = 1.65*980*((receptor_array)/10000)*0.0419
        else:
            # taucrit without areceptor
            taucrit = 1.65*980*0.0419
        
    elif plotvar == 'VEL':
        # Define critical velocity for motility as 0.05 m/s
        velcrit = 0.05 *np.ones(np.shape(np.squeeze(u[0,0,:,:])))    
        

    # Load and parse run order file. This csv file has the wave conditions for each case. The wave conditions are listed in the order of cases as they are 
    # stored in the first dimension of data_wecs or data_bs
    dev_present_dir = os.path.dirname(dev_present_file)
    
    foo = np.loadtxt(run_order_file,
                     delimiter=',', dtype='str')
    bc_name_wecs = foo[:,1]

    # Load BC file with probabilities and find appropriate probability
    BC_Annie = np.loadtxt(bc_file, delimiter=',', skiprows=1)

    #==============================
    # Load WECs NetCDF file without wecs into variable data_bs

    # Find name of NetCDF device not present output file
    # Read NetCDF file and parse contents needed for plotting
    file = Dataset(dev_notpresent_file)
    if plotvar != 'VEL':
        data_bs = file.variables[plotvar][:]
    else:
        u = file.variables['U1'][:,:,-1,:,:]
        v = file.variables['V1'][:,:,-1,:,:]
        
        data_bs = np.sqrt(u**2 + v**2)  
        
        
    file.close()
    
    wec_diff_bs = np.zeros(np.shape(data_bs[0,0,:,:]))
    wec_diff_wecs = np.zeros(np.shape(data_wecs[0,0,:,:]))
    # wec_diff = np.zeros(np.shape(data_wecs[0,0,:,:]))

    #=======================================================
    # set up a dataframe of probabilities
    df = pd.DataFrame({'Hs':BC_Annie[:,0].astype(float), 'Tp':BC_Annie[:,1].astype(int).astype(str), 'Dir':BC_Annie[:,2].astype(int).astype(str), 'prob':BC_Annie[:,4].astype(float)/100.})
    # generate a primary key for merge with the run order
    df['pk'] = ['Hs'] + df['Hs'].map('{:.2f}'.format) + ['Tp'] + df['Tp'].str.pad(2, fillchar = '0') + ['Dir'] + df['Dir']
    # set up a run order dataframe
    df_ro = pd.DataFrame({'bc_name': np.char.strip(bc_name_wecs), 'bcnum': foo[:,0].astype(int) - 1})
    # merge to the run order. This trims out runs that we want dropped.
    df_merge = pd.merge(df_ro, df, how = 'left', left_on = 'bc_name', right_on = 'pk')

    # currently we are missing a few runs so trim them out of the data_wecs
    df_merge = df_merge.loc[df_merge['bcnum'].values < data_wecs.shape[0], :]
    
    # Loop through all boundary conditions and create images
    for bcnum, prob in zip(df_merge['bcnum'], df_merge['prob']):
        
        #===============================================================
        # Compute normalized difference between with WEC and without WEC
        
        #wec_diff = wec_diff + prob*(data_w_wecs[bcnum,1,:,:] - data_wo_wecs[bcnum,1,:,:])/data_wo_wecs[bcnum,1,:,:]
        if plotvar == 'TAUMAX':
            # if the shapes are the same then process. Otherwise, process to an array and stop
            if data_bs[bcnum,-1,:,:].shape == taucrit.shape:
                wec_diff_bs = wec_diff_bs + prob*data_bs[bcnum,-1,:,:]/(taucrit*10)
                wec_diff_wecs = wec_diff_wecs + prob*data_wecs[bcnum,-1,:,:]/(taucrit*10)
            else:
                newarray=np.transpose(data_bs[bcnum,-1,:,:].data)
                array2 = np.flip(newarray, axis=0) 
                numpy_array = array2 
                rows, cols = numpy_array.shape
                # will need to dump to raster to check
                # will error as output path is not defined.
                SPATIAL_REFERENCE_SYSTEM_WKID = 4326 #WGS84 meters
                nbands = 1
                # bottom left, x, y netcdf file
                
                # from Kaus -235.8+360 degrees = 124.2 degrees. The 235.8 degree conventions follows longitudes that increase 
                # eastward from Greenwich around the globe. The 124.2W, or -124.2 goes from 0 to 180 degrees to the east of Greenwich.  
                bounds = [xcor.min() - 360,ycor.min()] #x,y or lon,lat, this is pulled from an input data source
                # look for dx/dy
                dx = xcor[1,0] - xcor[0,0]
                dy = ycor[0,1] - ycor[0,0]
                cell_resolution = [dx,dy ] #x res, y res or lon, lat, same as above
                
                output_raster = create_raster(output_path,
                      cols,
                      rows,
                      nbands)
                
                output_raster = numpy_array_to_raster(output_raster,
                              numpy_array,
                              bounds,
                              cell_resolution,
                              SPATIAL_REFERENCE_SYSTEM_WKID, output_path)
                
            
        elif plotvar == 'VEL':
            # breakpoint()
            # wec_diff_bs = wec_diff_bs + prob*(2*data_bs[bcnum,1,:,:] - data_bs[bcnum,1,:,:])/(velcrit*10)
            # wec_diff_wecs = wec_diff_wecs + prob*(2*data_wecs[bcnum,1,:,:] - data_wecs[bcnum,1,:,:])/(velcrit*10)
            # Should this be 0?
            wec_diff_bs = wec_diff_bs + prob*(2*data_bs[bcnum,0,:,:] - data_bs[bcnum,0,:,:])/(velcrit*10)
            wec_diff_wecs = wec_diff_wecs + prob*(2*data_wecs[bcnum,0,:,:] - data_wecs[bcnum,0,:,:])/(velcrit*10)
            
        elif plotvar == 'DPS':
            wec_diff_bs = wec_diff_bs + prob*data_bs[bcnum,1,:,:]
            wec_diff_wecs = wec_diff_wecs + prob*data_wecs[bcnum,1,:,:]
            #wec_diff = (data_w_wecs[bcnum,1,:,:] - data_wo_wecs[bcnum,1,:,:])/data_wo_wecs[bcnum,1,:,:]
    #========================================================


    # Calculate risk metrics over all runs
    if plotvar == 'TAUMAX' or plotvar == 'VEL':
        wec_diff_bs_sgn = np.floor(wec_diff_bs*25)/25 
        wec_diff_wecs_sgn = np.floor(wec_diff_wecs*25)/25 

        wec_diff = (np.sign(wec_diff_wecs_sgn-wec_diff_bs_sgn)*wec_diff_wecs_sgn) 
        wec_diff = wec_diff.astype(int) + wec_diff_wecs-wec_diff_bs
        
        # set to zero
        #wec_diff[np.abs(wec_diff)<0.01] = 0
        
    elif plotvar == 'DPS':
        wec_diff =  wec_diff_wecs - wec_diff_bs
        wec_diff[np.abs(wec_diff)<0.0005] = 0
        
    #========================================================
        
    #convert to a geotiff, using wec_diff 

    #listOfFiles = [wec_diff_bs, wec_diff_wecs, wec_diff, wec_diff_bs_sgn, wec_diff_wecs_sgn]  

    #transpose and pull

    newarray=np.transpose(wec_diff)
    array2 = np.flip(newarray, axis=0) 
    rows, cols = array2.shape

    #get number of rows and cols
    return(rows, cols, array2)    
        
def calculate_diff_cec(fname_base, fname_cec, taucrit=100.):

    """
    Given non linear grid files calculate the difference. Currently taucrit is 100. Would need to make
    constant raster of from either fname_base or fname_cec for raster use.
    """
    
    f_base = Dataset(fname_base, mode='r', format='NETCDF4')
    f_cec = Dataset(fname_cec, mode='r', format='NETCDF4')
    
    tau_base = f_base.variables['taus'][1, :].data
    tau_cec = f_cec.variables['taus'][1, :].data

    lon = f_base.variables['FlowElem_xcc'][:].data
    lat = f_base.variables['FlowElem_ycc'][:].data

    #lon, lat = transformer.transform(f_base.variables['NetNode_x'][:].data, f_base.variables['NetNode_y'][:].data)
    # df = pd.DataFrame({'lon': lon, 'lat':lat})
    # df.to_csv('out_test_lon_lat.csv', index = False)

    wec_diff_bs = np.zeros(np.shape(tau_base))
    wec_diff_wecs = np.zeros(np.shape(tau_base))
    wec_diff = np.zeros(np.shape(tau_base))

    #taucrit = 1.65*980*((1.9e-4*10**6)/10000)*0.0419
    taucrit = 100.
    return_interval = 1
    
    prob = 1/return_interval
    wec_diff_bs = wec_diff_bs + prob*tau_base/(taucrit*10)
    wec_diff_wecs = wec_diff_wecs + prob*tau_cec/(taucrit*10)

    wec_diff_bs_sgn = np.floor(wec_diff_bs*25)/25 
    wec_diff_wecs_sgn = np.floor(wec_diff_wecs*25)/25 
    wec_diff = (np.sign(wec_diff_wecs_sgn-wec_diff_bs_sgn)*wec_diff_wecs_sgn) 
    wec_diff = wec_diff.astype(int) + wec_diff_wecs-wec_diff_bs   
    wec_diff[np.abs(wec_diff)<0.001] = 0

    # Use triangular interpolation to generate grid
    # reflon=np.linspace(lon.min(),lon.max(),1000)
    # reflat=np.linspace(lat.min(),lat.max(),1000)
    reflon=np.linspace(lon.min(),lon.max(),169)
    reflat=np.linspace(lat.min(),lat.max(),74)

    reflon,reflat=np.meshgrid(reflon,reflat)
 
    # original
    flatness=0.1  # flatness is from 0-.5 .5 is equilateral triangle
    flatness=0.2  # flatness is from 0-.5 .5 is equilateral triangle
    tri=Triangulation(lon,lat)
    mask = TriAnalyzer(tri).get_flat_tri_mask(flatness)
    tri.set_mask(mask)
    tli=LinearTriInterpolator(tri,wec_diff)
    tau_interp=tli(reflon,reflat)

    newarray=np.transpose(tau_interp[:, :].data)
    array2 = np.flip(tau_interp[:, :].data, axis=0) 

    # rows, cols = np.shape(tau_interp)
    rows, cols = np.shape(array2)
    
    return (rows, cols, array2)
#returns gdal data source raster object
def create_raster(output_path,
                  cols,
                  rows,
                  nbands):
    # create gdal driver - doing this explicitly
    driver = gdal.GetDriverByName(str('GTiff'))

    output_raster = driver.Create(output_path,
                                  int(cols),
                                  int(rows),
                                  nbands,
                                  eType = gdal.GDT_Float32)  
    
    # spatial_reference = osr.SpatialReference()
    # spatial_reference.ImportFromEPSG(spatial_reference_system_wkid)
    # output_raster.SetProjection(spatial_reference.ExportToWkt())
    
    return output_raster

#returns a gdal raster data source
def numpy_array_to_raster(output_raster,
                          numpy_array,
                          bounds,
                          cell_resolution,
                          spatial_reference_system_wkid, output_path):
   
# create output raster
#(upper_left_x, x_resolution, x_skew 0, upper_left_y, y_skew 0, y_resolution). 
# Need to rotate to go from np array to geo tiff. This can vary depending on the methods used above. Will need to test for this.
    geotransform = (bounds[0],
                    cell_resolution[0],
                    0,
                    bounds[1] + cell_resolution[1],
                    0,
                    -1 * cell_resolution[1])

    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromEPSG(spatial_reference_system_wkid)

    output_raster.SetProjection(spatial_reference.ExportToWkt()) #exports the cords to the file
    output_raster.SetGeoTransform(geotransform)
    output_band = output_raster.GetRasterBand(1)
    #output_band.SetNoDataValue(no_data) #Not an issue, may be in other cases?
    output_band.WriteArray(numpy_array)

    output_band.FlushCache()
    output_band.ComputeStatistics(False) #you want this false, true will make computed results, but is faster, could be a setting in the UI perhaps, esp for large rasters?
    
    if os.path.exists(output_path) == False:
        raise Exception('Failed to create raster: %s' % output_path)  
        
    output_raster = None
    return output_path

#now call the functions
if __name__ == "__main__":

    #=================
    # User input block
    
    # Set directory with output folders (contains with_wecs and without_wecs folders)
    # run_dir = r'C:\Users\mjamieson61\Documents\Internal_Working\Projects\QGIS_Python\Codebase'
    # linux
    
    # Set plot variable
    # plotvar = 'VEL'      # Concentrations per layer at zeta point
    plotvar = 'TAUMAX'  # Tau max in zeta points (N/m^2)
    #plotvar = 'DPS'     # Bottom depth at zeta point (m)
    
    # Set NetCDF file to load WEC
    dev_present_file = r"C:\Users\ependleton52\Desktop\temp_local\QGIS\Code_Model\Codebase\run_dir_wecs\trim_sets_flow_inset_allruns.nc"
    dev_notpresent_file = r"C:\Users\ependleton52\Desktop\temp_local\QGIS\Code_Model\Codebase\run_dir_nowecs\trim_sets_flow_inset_allruns.nc"
    
    # cec files
    # dev_present_file = r"C:\Users\ependleton52\Desktop\temp_local\QGIS\Code_Model\Codebase\cec\with_cec_1.nc"
    # dev_notpresent_file = r"C:\Users\ependleton52\Desktop\temp_local\QGIS\Code_Model\Codebase\cec\no_cec_1.nc"
    
    # dev_present_file = r"C:\Users\ependleton52\Desktop\temp_local\QGIS\Code_Model\Codebase\tanana\DFM_OUTPUT_tanana100\modified\tanana100_map_0_tanana1_cec.nc"
    # dev_notpresent_file = r"C:\Users\ependleton52\Desktop\temp_local\QGIS\Code_Model\Codebase\tanana\DFM_OUTPUT_tanana100\modified\tanana100_map_6_tanana1_cec.nc"
        
    # set the boundary_coditions file
    bc_file = r"C:\Users\ependleton52\Desktop\temp_local\QGIS\Code_Model\Codebase\BC_Annie_Annual_SETS.csv"
    
    # run order file
    run_order_file = r"C:\Users\ependleton52\Desktop\temp_local\QGIS\Code_Model\Codebase\run_dir_wecs\run_order_wecs.csv"
    
    # configuration for raster translate
    GDAL_DATA_TYPE = gdal.GDT_Float32 
    GEOTIFF_DRIVER_NAME = r'GTiff'
    
    # Skip the bad runs for now
    bcarray = np.array([0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,19,20,22])
    
    # all runs
    # bcarray = [i for i in range(1,23)]
    
    #SWAN will always be in meters. Not always WGS84
    SPATIAL_REFERENCE_SYSTEM_WKID = 4326 #WGS84 meters
    nbands = 1 #should be one always right?
    # bounds = [-124.2843933,44.6705] #x,y or lon,lat, this is pulled from an input data source
    # cell_resolution = [0.0008,0.001 ] #x res, y res or lon, lat, same as above

    # from Kaus -235.8+360 degrees = 124.2 degrees. The 235.8 degree conventions follows longitudes that increase 
    # eastward from Greenwich around the globe. The 124.2W, or -124.2 goes from 0 to 180 degrees to the east of Greenwich.  
    bounds = [xcor.min() - 360,ycor.min()] #x,y or lon,lat, this is pulled from an input data source
    # look for dx/dy
    dx = xcor[1,0] - xcor[0,0]
    dy = ycor[0,1] - ycor[0,0]
    cell_resolution = [dx,dy ]


    #will we ever need to do a np.isnan test?
    #NO_DATA = 'NaN'

    #set output path
    output_path = r'C:\Users\ependleton52\Desktop\temp_local\QGIS\Code_Model\Codebase\rasters\rasters_created\plugin\out_calculated.tif'
    
    # Functions
    rows, cols, numpy_array = transform_netcdf(dev_present_file, dev_notpresent_file, bc_file, run_order_file, bcarray, plotvar)
    
    
    output_raster = create_raster(output_path,
                      cols,
                      rows,
                      nbands)
 
    # post processing of numpy array to output raster
    output_raster = numpy_array_to_raster(output_raster,
                              numpy_array,
                              bounds,
                              cell_resolution,
                              SPATIAL_REFERENCE_SYSTEM_WKID, output_path)
    
    """    
    # add the raster to the QGIS interface
    #newRLayer = iface.addRasterLayer(output_raster)
    
    #now add to the interface of QGIS
    #newRLayer = iface.addRasterLayer(output_path)
    """