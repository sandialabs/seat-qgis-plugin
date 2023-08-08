from scipy import interpolate
import numpy as np
from scipy.signal import hilbert, stft
import xarray as xr

def init_wfs():
    # No params, initializes the weighting function xarrays
    # Output is dictionary of xarrays corresponding of weighting functions, in dB, f is in Hz
    freq_np = np.arange(0.01, 1000, 0.01) # in kHz

    # Low-frequency cetaceans  weighting function
    # citation: NMFS 2018
    a = 1
    b = 2
    f1 = 0.2
    f2 = 19
    C = 0.13
    W_LF = C+ 10*np.log10(((freq_np/f1)**(2*a))/(((1+(freq_np/f1)**2)**a)*((1+(freq_np/f2)**2)**b)))
    W_LF = xr.DataArray(W_LF, coords={'f': freq_np*1000}) # in Hz

    # Mid frequency weighting function
    # citation: NMFS 2018
    a = 1.6
    b = 2
    f1 = 8.8
    f2 = 110
    C = 1.20
    W_MF = C+ 10*np.log10(((freq_np/f1)**(2*a))/(((1+(freq_np/f1)**2)**a)*((1+(freq_np/f2)**2)**b)))
    W_MF = xr.DataArray(W_MF, coords={'f': freq_np*1000})   # in Hz

    # High frequency weighting function
    # citation: NMFS 2018       
    a = 1.8
    b = 2
    f1 = 12
    f2 = 140
    C = 1.36
    W_HF = C+ 10*np.log10(((freq_np/f1)**(2*a))/(((1+(freq_np/f1)**2)**a)*((1+(freq_np/f2)**2)**b)))
    W_HF = xr.DataArray(W_HF, coords={'f': freq_np*1000})  # in Hz

    # Phocid pinnipeds weighting function
    # citation: NMFS 2018
    a = 1
    b = 2
    f1 = 1.9
    f2 = 30
    C = 0.75
    W_PW = C+ 10*np.log10(((freq_np/f1)**(2*a))/(((1+(freq_np/f1)**2)**a)*((1+(freq_np/f2)**2)**b)))
    W_PW = xr.DataArray(W_PW, coords={'f': freq_np*1000})  # in Hz

    # Otariid pinnipeds weighting function
    # citation: NMFS 2018
    a = 2
    b = 2
    f1 = 0.94
    f2 = 25
    C = 0.64
    W_OW = C+ 10*np.log10(((freq_np/f1)**(2*a))/(((1+(freq_np/f1)**2)**a)*((1+(freq_np/f2)**2)**b)))
    W_OW = xr.DataArray(W_OW, coords={'f': freq_np*1000})  # in Hz

    weighting_functions = {"LFC":W_LF, "MFC":W_MF, "HFC":W_HF, "PPW":W_PW, "OPW":W_OW}
    return(weighting_functions)


# ==========================================================================
# Peak sound pressure level (SPL_pk)
# ==========================================================================


def SPL_pk_dB(y):
    # Equaltion 2 from Wilfod et al 2021
    # Assumes the value is in same units as reference value [uPa for pressure, nm/s for velocity]
    # y = originally sampled time series
    # Output is in dB re 1 uPa for pressure or re 1 nm/s for velocity
    if len(y) > 0:
        SPL_pk = 10 * np.log10(np.max(y**2))
        return float(SPL_pk)
    else:
        return np.nan


# ==========================================================================
# Root mean square Sound Pressure Level (SPL_rms)
# ==========================================================================


def SPL_rms_dB(y, dt):
    # Equaltion 1 from Wilfod et al 2021
    # Assumes the value of y is in same units as reference value [uPa for pressure, nm/s for velocity]
    # y = originally sampled time series
    # dt = time step of originally sampled time series (in s)
    # Output is in dB re 1 uPa or re 1 nm/s
    if len(y) > 0:
        T = len(y) * dt
        rms_SPL_dB = 20 * np.log10(np.sqrt((1 / T) * np.trapz(y**2, dx=dt)))
        return rms_SPL_dB
    else:
        return np.nan


# ==========================================================================
# Sound Exposure Level (SEL)
# ==========================================================================


def SEL_dB(y, dt):
    # Assumes the value of y is in same units as reference value [uPa for pressure, nm/s for velocity]
    # y = originally sampled time series
    # dt = time step of originally sampled time series (in s)
    # Output is in dB re 1 uPa * s  or dB re 1 nm/s * s
    if len(y) > 0:
        SEL = 10 * np.log10(np.trapz(y**2, dx=dt))
        return SEL
    else:
        return np.nan
    
def inverse_fft(f, Xf, T):
    """
    Parameters:
    f: array of frequencies at which Xf is defined
    Xf: spectrum defined at frequencies in f (in uPa), must be same length as f
    T: ending time stamp at which inverse FFT is desired, generally 1 second

    Outputs:
    x: array of pressure time series, in same units as Xf
    dt: time step of x, in seconds
    """
    fmax = f.max()
    df = 1/T
    fs = int(fmax*2)
    Nfft = int(fs * T)

    fnew = np.arange(0, int(fs/2)+df, df)
    func = interpolate.interp1d(f, Xf, bounds_error=False, fill_value=0)

    Xnew = func(fnew)
    Xfull = np.concatenate((Xnew[0:-1], np.flip(Xnew[1:])))
 
    x = np.real(np.fft.ifft(Xfull, Nfft))
    dt = 1/fs

    return x, dt
# ==========================================================================
# Calculate all metrics from original frequency spectrum; transmission loss dictionary, and weighting functions
# ==========================================================================
def calc_all_metrics(f_og_db, w_fxn):
    """
    Parameters:
    f_og_db = frequency xarray of original frequency domain in dB; one dimension must be called f (frequency, in Hz).  1 Hz resolution, from 0-1023 Hz. 
    w_fxn = weighting function xarray by frequency; one dimension must be called f (frequency, in Hz), higher resolution and coverage than f_og_db is better   
    
    Outputs:
    returns dictionary of metrics of rmsSPL, peakSPL, SEL, both weighted and unweighted, all in dB
    """
    # Reshape appropriate weighting function to match same shape as f_og_db
    w_fxn_sub = w_fxn.sel(f=f_og_db['f'],  method = "nearest")

    # Calculate frequency resolution of transmission loss 
    # freqs = [int(x.replace("TL", "")) for x in list(TL_dict.keys())]
    # freqs.sort()
    # freq_resolution = freqs[1]- freqs[0]

    # # Produce modified spectra at point of interest (f_mod_dB) based on tranmission loss step function
    # f_mod_db = f_og_db.copy()
    # for key in TL_dict.keys():
    #     freq = int(key.replace("TL", ""))
    #     f1, f2 = int(freq-freq_resolution/2), int(freq+freq_resolution/2)
    #     TL = TL_dict[key] # in dB
    #     f_mod_db[f1:f2] = f_og_db[f1:f2]-TL
    
    # Apply weighting functions to modified spectra
    f_mod_db_w = xr.DataArray(w_fxn_sub.values + f_og_db.values, coords={'f': f_og_db['f']}) 

    # Turn f_mod and f_mod_w into uPa from dB
    # f_mod_w = 10**(f_mod_db_w.real/20)
    f_mod = 10**(f_og_db.real/20)

    # # inverse fft to switch into time domain for both the modified and weighted modified spectra
    # # f_mod is array of uPa from 0-1023 Hz at 1 Hz intervals
    fs = 2048 # sampling frequency, in Hz
    # # get time series of flat modified spectrum
    # f_mod_double = np.concatenate((f_mod, np.flipud(f_mod)))
    # t_mod = np.real(np.fft.ifft(f_mod_double))
    # time_coords = np.arange(0,len(t_mod))/fs
    # t_mod_xr = xr.DataArray(t_mod, coords={'time': time_coords}) 

    # # get time series of weighted modified spectrum
    # f_mod_double_w = np.concatenate((f_mod_w, np.flipud(f_mod_w)))
    # t_mod_w = np.real(np.fft.ifft(f_mod_double_w))
    # t_mod_xr_w = xr.DataArray(t_mod_w, coords={'time': time_coords}) 
    t_mod_xr, dt = inverse_fft(f_mod.f, f_mod, 1)
    t_mod_xr_w = xr.DataArray(t_mod_xr, coords={'time': np.arange(0,1,dt)}) 

    dt = 1/fs # time step, in s
    metrics = {}
    metrics['SPL_rms_flat'] = SPL_rms_dB(t_mod_xr, dt)
    metrics['SPL_rms_weighted'] = SPL_rms_dB(t_mod_xr_w, dt)
    metrics['SPL_pk_flat'] = SPL_pk_dB(t_mod_xr)
    metrics['SPL_pk_weighted'] = SPL_pk_dB(t_mod_xr_w)
    metrics['SEL_flat'] = SEL_dB(t_mod_xr, dt)
    metrics['SEL_weighted'] = SEL_dB(t_mod_xr_w, dt)

    return metrics

def convert_dataset(filename):
    # Simplifies Dataset structure for processing
    DS = xr.open_dataset(filename)
    data_set = xr.Dataset()
    # data_set.coords['LAT'] = (("lat","lon"), DS.YCOR.to_numpy())
    # data_set.coords['LON'] = (("lat", "lon"), DS.XCOR.to_numpy())
    data_set.coords['lat'] = DS.YCOR.to_numpy()[:,0]
    data_set.coords['lon'] = DS.XCOR.to_numpy()[0,:]
    data_set.coords['z'] = DS.ZCOR.to_numpy()
    data_set.coords['f'] = DS.Fc.to_numpy()
    data_set = data_set.assign(totSPL=(['z', 'lat','lon'], DS.totSPL.to_numpy()))
    data_set = data_set.assign(octSPL=(['z', 'lat','lon' ,'f'], DS.octSPL.to_numpy()))

    data_set['lat'].attrs = DS.YCOR.attrs
    data_set['lon'].attrs = DS.XCOR.attrs
    data_set['z'].attrs = DS.ZCOR.attrs
    data_set['f'].attrs = DS.Fc.attrs
    data_set['totSPL'].attrs = DS.totSPL.attrs
    data_set['octSPL'].attrs = DS.octSPL.attrs


    # data_set.to_netcdf(os.path.join(outpath, os.path.basename(filename)))
    return data_set

def calc_stressor_paracousti(paracousti_folder,
                             paracousti_files,
                             receptor_file):
    from tqdm import tqdm, trange
    import pandas as pd
    import os
    import xarray as xr
    import numpy as np
    receptor = pd.read_csv(receptor_file, index_col=0, header=None).T
    WFX = init_wfs()
    f_new = np.arange(0,1024)
    for ip, pfile in enumerate(paracousti_files):
        print(f"{ip}/{len(paracousti_files)}")
        DS = convert_dataset(os.path.join(paracousti_folder, pfile))
        SPL_interp = DS.octSPL.interp(f=f_new)
        SPL_interp = xr.where(np.isnan(SPL_interp), 0, SPL_interp)
        if ip==0:
            shape_out = (len(paracousti_files), len(SPL_interp.lat), len(SPL_interp.lon), len(SPL_interp.z))
            Metrics = {'SPL_rms_flat': np.zeros(shape=shape_out),
            'SPL_rms_weighted':  np.zeros(shape=shape_out),
            'SPL_pk_flat':  np.zeros(shape=shape_out),
            'SPL_pk_weighted':  np.zeros(shape=shape_out),
            'SEL_flat':  np.zeros(shape=shape_out),
            'SEL_weighted':  np.zeros(shape=shape_out)}
        for ilat in trange(len(SPL_interp.lat)):
            for ilon in range(len(SPL_interp.lon)):
                for iz in range(len(SPL_interp.z)):
                    metric = calc_all_metrics(f_og_db=SPL_interp.isel(lat=ilat, lon=ilon, z=iz), w_fxn=WFX[receptor.Marine_Group.to_numpy().item()])
                    for key in metric.keys():
                        Metrics[key][ip, ilat,ilon,iz] = metric[key]

    Metrics_Depth_Average = {k:[] for k in Metrics.keys()}
    Metrics_Depth_Max = {k:[] for k in Metrics.keys()}
    for key in Metrics.keys():
        Metrics_Depth_Average[key] = np.nanmean(Metrics[key], axis=3)
        Metrics_Depth_Max[key] = np.nanmax(Metrics[key], axis=3)
    Metrics['lat'] = Metrics_Depth_Average['lat'] = Metrics_Depth_Max['lat'] = SPL_interp.lat.values
    Metrics['lon'] = Metrics_Depth_Average['lat'] = Metrics_Depth_Max['lat'] = SPL_interp.lon.values
    return Metrics, Metrics_Depth_Average, Metrics_Depth_Max