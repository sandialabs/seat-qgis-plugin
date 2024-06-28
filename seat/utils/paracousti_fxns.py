import numpy as np
import xarray as xr
from scipy import interpolate
import os
import warnings
from tqdm import tqdm

warnings.simplefilter("ignore")


def apply_Gf(a, b, f1, f2, C, freq_np=None):
    # Weighting function Finneran (2015)
    # see Criteria and Thresholds for U.S. Navy Acoustic and Explosive Effects Analysis (Phase III)
    # https://nwtteis.com/portals/nwtteis/files/technical_reports/Criteria_and_Thresholds_for_U.S._Navy_Acoustic_and_Explosive_Effects_Analysis_June2017.pdf
    # (EQ1 and Table 2-8)
    if freq_np is None:
        freq_np = np.arange(0.001, 1000, 0.01)  # in kHz
    numer = (freq_np / f1) ** (2 * a)
    denom = ((1 + (freq_np / f1) ** 2) ** a) * ((1 + (freq_np / f2) ** 2) ** b)
    # Gf = C + 10*np.log10(((freq_np/f1)**(2*a))/(((1+(freq_np/f1)**2)**a)*((1+(freq_np/f2)**2)**b)))
    Gf = C + 10 * np.log10(numer / denom)
    Gf = xr.DataArray(Gf, coords={"f": freq_np * 1000})  # in Hz
    return Gf


def in_water_weights():
    # No params, initializes the weighting function xarrays
    # Output is dictionary of xarrays corresponding of weighting functions, in dB, f is in Hz
    freq_np = np.arange(0.001, 1000, 0.01)  # in kHz

    # Low frequency weighting function
    # citation: NMFS 2018
    a = 1
    b = 2
    f1 = 0.2
    f2 = 19
    C = 0.13
    W_LF = apply_Gf(a, b, f1, f2, C, freq_np)
    # W_LF = C+ 10*np.log10(((freq_np/f1)**(2*a))/(((1+(freq_np/f1)**2)**a)*((1+(freq_np/f2)**2)**b)))
    # W_LF = xr.DataArray(W_LF, coords={'f': freq_np*1000}) # in Hz

    # Mid frequency weighting function
    # citation: NMFS 2018
    a = 1.6
    b = 2
    f1 = 8.8
    f2 = 110
    C = 1.20
    W_MF = apply_Gf(a, b, f1, f2, C, freq_np)
    # W_MF = C+ 10*np.log10(((freq_np/f1)**(2*a))/(((1+(freq_np/f1)**2)**a)*((1+(freq_np/f2)**2)**b)))
    # W_MF = xr.DataArray(W_MF, coords={'f': freq_np*1000})   # in Hz

    # High frequency weighting function
    # citation: NMFS 2018
    a = 1.8
    b = 2
    f1 = 12
    f2 = 140
    C = 1.36
    W_HF = apply_Gf(a, b, f1, f2, C, freq_np)
    # W_HF = C+ 10*np.log10(((freq_np/f1)**(2*a))/(((1+(freq_np/f1)**2)**a)*((1+(freq_np/f2)**2)**b)))
    # W_HF = xr.DataArray(W_HF, coords={'f': freq_np*1000})  # in Hz

    # Phocid pinnipeds weighting function
    # citation: NMFS 2018
    a = 1
    b = 2
    f1 = 1.9
    f2 = 30
    C = 0.75
    W_PW = apply_Gf(a, b, f1, f2, C, freq_np)
    # W_PW = C+ 10*np.log10(((freq_np/f1)**(2*a))/(((1+(freq_np/f1)**2)**a)*((1+(freq_np/f2)**2)**b)))
    # W_PW = xr.DataArray(W_PW, coords={'f': freq_np*1000})  # in Hz

    # Otariid pinnipeds weighting function
    # citation: NMFS 2018
    a = 2
    b = 2
    f1 = 0.94
    f2 = 25
    C = 0.64
    W_OW = apply_Gf(a, b, f1, f2, C, freq_np)
    # W_OW = C+ 10*np.log10(((freq_np/f1)**(2*a))/(((1+(freq_np/f1)**2)**a)*((1+(freq_np/f2)**2)**b)))
    # W_OW = xr.DataArray(W_OW, coords={'f': freq_np*1000})  # in Hz

    # Turtles
    # citation (Finneran 2016)
    a = 1.4
    b = 2
    f1 = 0.077
    f2 = 0.440
    C = 2.35
    W_TU = apply_Gf(a, b, f1, f2, C, freq_np)
    # W_TU = C+ 10*np.log10(((freq_np/f1)**(2*a))/(((1+(freq_np/f1)**2)**a)*((1+(freq_np/f2)**2)**b)))
    # W_TU = xr.DataArray(W_TU, coords={'f': freq_np*1000})  # in Hz

    # Manatee
    # Criteria_and_Thresholds_for_U.S._Navy_Acoustic_and_Explosive_Effects_Analysis_June2017.pdf
    # https://nwtteis.com/portals/nwtteis/files/technical_reports/Criteria_and_Thresholds_for_U.S._Navy_Acoustic_and_Explosive_Effects_Analysis_June2017.pdf
    #
    a = 1.8
    b = 2
    f1 = 4.3
    f2 = 25
    C = 2.62
    W_SI = apply_Gf(a, b, f1, f2, C, freq_np)

    weighting_functions = {
        "LFC": W_LF,
        "MFC": W_MF,
        "HFC": W_HF,
        "PPW": W_PW,
        "OW": W_OW,
        "TU": W_TU,
        "SI": W_SI,
    }  # , "OA":W_OA, "PA":W_PA}
    return weighting_functions


def in_air_weights():
    # No params, initializes the weighting function xarrays
    # Output is dictionary of xarrays corresponding of weighting functions, in dB, f is in Hz
    freq_np = np.arange(0.001, 1000, 0.01)  # in kHz

    # OA, e.g. Callorhinus ursinus, Zalophus californianus (northern fur sea, california sea lion)
    # Criteria_and_Thresholds_for_U.S._Navy_Acoustic_and_Explosive_Effects_Analysis_June2017.pdf
    # https://nwtteis.com/portals/nwtteis/files/technical_reports/Criteria_and_Thresholds_for_U.S._Navy_Acoustic_and_Explosive_Effects_Analysis_June2017.pdf
    a = 1.4
    b = 2
    f1 = 2
    f2 = 20
    C = 1.39
    W_OA = apply_Gf(a, b, f1, f2, C, freq_np)

    # PA, Phoca vitulina (harbor seal, common seal)
    # Criteria_and_Thresholds_for_U.S._Navy_Acoustic_and_Explosive_Effects_Analysis_June2017.pdf
    # https://nwtteis.com/portals/nwtteis/files/technical_reports/Criteria_and_Thresholds_for_U.S._Navy_Acoustic_and_Explosive_Effects_Analysis_June2017.pdf
    a = 2
    b = 2
    f1 = 0.75
    f2 = 8.3
    C = 1.50
    W_PA = apply_Gf(a, b, f1, f2, C, freq_np)
    weighting_functions = {"OA": W_OA, "PA": W_PA}
    return weighting_functions


"""

    plt.close()
    for key in weighting_functions.keys():
        plt.plot(weighting_functions[key].f, weighting_functions[key])
    plt.xscale('log',base=10)
    ax = plt.gca()
    from matplotlib import ticker
    # ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.xaxis.set_minor_locator(ticker.NullLocator())  # no minor ticks
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())  # set regular formatting
    ax.ticklabel_format(style='sci', scilimits=(-6, 9))  # disable scientific notation
    ax.set_ylim(-60,0)   
    plt.show()
"""


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


# ==========================================================================
# Inverse FFT
# ==========================================================================


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
    df = 1 / T
    fs = int(fmax * 2)
    Nfft = int(fs * T)

    fnew = np.arange(0, int(fs / 2) + df, df)
    func = interpolate.interp1d(f, Xf, bounds_error=False, fill_value=0)

    Xnew = func(fnew)
    Xfull = np.concatenate((Xnew[0:-1], np.flip(Xnew[1:])))

    x = np.real(np.fft.ifft(Xfull, Nfft))
    dt = 1 / fs

    return x, dt


def calc_SPL_SEL_metrics(psif, w_fxn=None):
    """
    Calculates all metrics for output of PE model.
    Parameters:
    f_og_db = frequency xarray of spectra at a single range/depth point; one dimension must be called f (frequency, in Hz).  1 Hz resolution, from 0-1023 Hz.
    w_fxn = weighting function xarray by frequency; one dimension must be called f (frequency, in Hz), higher resolution and coverage than f_og_db is better

    Outputs:
    returns dictionary of metrics of rmsSPL, peakSPL, SEL, both weighted and unweighted, all in dB
    """
    faxis = psif["f"].values

    nf = len(faxis)
    fc = (faxis[0] + faxis[-1]) / 2
    nyqst = int(np.ceil((nf + 1) / 2))
    # bw = (faxis[-1] - faxis[0])/2
    # wind = np.sinc((faxis - fc)/bw)
    wind = np.ones(
        len(faxis)
    )  # Use rectangular window instead since we have a frequency-dependent source function
    fs = 4 * fc
    T = 1
    N = int(fs * T)
    # tdelay = taxis[0,:]
    if w_fxn is None:
        psif_w = psif
    else:
        # Reshape appropriate weighting function to match same shape as f_og_db
        w_fxn_sub = w_fxn.sel(f=psif["f"], method="nearest")

        # Apply weighting functions to modified spectra
        newmag = 10 ** ((w_fxn_sub.values + 20 * np.log10(np.abs(psif.values))) / 20)
        psif_w = xr.DataArray(
            psif.values * newmag / np.abs(psif.values), coords={"f": psif["f"]}
        )

    data = wind * np.conj(psif_w.values)  # * np.exp(1j*2*np.pi*faxis*tdelay[100])[:,0]
    data = np.concatenate((data[nyqst : nf + 1], np.zeros(N - nf), data[0:nyqst]))
    psi_t_w = np.real(np.fft.ifft(data)) * np.sqrt(fs * N)
    psi_t_w_xr = xr.DataArray(psi_t_w, coords={"time": np.linspace(0, T, N)})

    dt = psi_t_w_xr["time"].values[1]

    metrics = {}
    if w_fxn is None:
        metrics["SPL_rms_flat"] = SPL_rms_dB(psi_t_w_xr, dt)
        metrics["SPL_pk_flat"] = SPL_pk_dB(psi_t_w_xr)
        metrics["SEL_flat"] = SEL_dB(psi_t_w_xr, dt)
    else:
        metrics["SPL_rms_weighted"] = SPL_rms_dB(psi_t_w_xr, dt)
        metrics["SPL_pk_weighted"] = SPL_pk_dB(psi_t_w_xr)
        metrics["SEL_weighted"] = SEL_dB(psi_t_w_xr, dt)

    return metrics


def metric_descriptions():
    Descriptions = {
        "SPL_rms_flat": "unweighted root-mean-square sound pressure level",
        "SPL_pk_flat": "unweighted peak sound pressure level",
        "SEL_flat": "unweighted sound exposure level",
        "LFC_SPL_rms_weighted": "low-frequency cetaceans root-mean-square sound pressure level",
        "LFC_SPL_pk_weighted": "low-frequency cetaceans peak sound pressure level",
        "LFC_SEL_weighted": "low-frequency cetaceans sound exposure level",
        "MFC_SPL_rms_weighted": "mid-frequency cetaceans root-mean-square sound pressure level",
        "MFC_SPL_pk_weighted": "mid-frequency cetaceans peak sound pressure level",
        "MFC_SEL_weighted": "mid-frequency cetaceans sound exposure level",
        "HFC_SPL_rms_weighted": "high-frequency cetaceans root-mean-square sound pressure level",
        "HFC_SPL_pk_weighted": "high-frequency cetaceans peak sound pressure level",
        "HFC_SEL_weighted": "high-frequency cetaceans sound exposure level",
        "PPW_SPL_rms_weighted": "phocid pinnipeds in water root-mean-square sound pressure level",
        "PPW_SPL_pk_weighted": "phocid pinnipeds in water peak sound pressure level",
        "PPW_SEL_weighted": "phocid pinnipeds in water sound exposure level",
        "OW_SPL_rms_weighted": "otariids/other marine carnivores in water root-mean-square sound pressure level",
        "OW_SPL_pk_weighted": "otariids/other marine carnivores in water peak sound pressure level",
        "OW_SEL_weighted": "otariids/other marine carnivores in water sound exposure level",
        "TU_SPL_rms_weighted": "sea turtles root-mean-square sound pressure level",
        "TU_SPL_pk_weighted": "sea turtles peak sound pressure level",
        "TU_SEL_weighted": "sea turtles sound exposure level",
        "SI_SPL_rms_weighted": "sirenians root-mean-square sound pressure level",
        "SI_SPL_pk_weighted": "sirenians peak sound pressure level",
        "SI_SEL_weighted": "sirenians sound exposure level",
    }
    return Descriptions


def calc_paracousti_metrics(paracousti_path, save_path, weights="All"):
    w_fxns = in_water_weights()
    var_descriptions = metric_descriptions()
    paracousti_files = [i for i in os.listdir(paracousti_path) if i.endswith(".nc")]

    for fix in tqdm(range(len(paracousti_files)), desc="Processing Paracousti Files"):
        # Load Dataset
        DS = xr.open_dataset(os.path.join(paracousti_path, paracousti_files[fix]))

        # Calculate pressure from SPL
        DS = DS.assign(
            press_muPa=(
                ["Nz", "Ny", "Nx", "fc"],
                20 * 10 ** (DS.octSPL.to_numpy() / 20),
                {"units": "micro Pascals"},
            )
        )
        # DS['press_muPa'] = 20 * 10**(DS.octSPL/20) #dB to microPascals

        # Reshape to single array to reduce loops
        orig_shape = DS["press_muPa"].shape
        press_f = (
            DS["press_muPa"]
            .to_numpy()
            .reshape(np.prod(orig_shape[0:3]), orig_shape[-1])
        )

        # Initialize Empty variables
        unweighted_vars = ["SPL_rms_flat", "SPL_pk_flat", "SEL_flat"]
        Unweighted_Dict = {}
        for unweighted_var in unweighted_vars:
            Unweighted_Dict[unweighted_var] = np.empty((np.shape(press_f)[0]))

        weighted_vars = ["SPL_rms_weighted", "SPL_pk_weighted", "SEL_weighted"]
        Weighted_Dict = {}
        for weight in list(w_fxns.keys()) if weights == "All" else weights:
            # print(weight)
            Weighted_Dict[weight] = {}
            for weighted_var in weighted_vars:
                Weighted_Dict[weight][weighted_var] = np.empty((np.shape(press_f)[0]))

        # create inputs for faster processing
        freqs = DS["Fc"].values
        pf = [
            xr.DataArray(press_f[pix, :], coords={"f": freqs})
            for pix in range(press_f.shape[0])
        ]

        # Calculate Metrics
        for pix in tqdm(
            range(press_f.shape[0]),
            desc=f"Calculating Metrics for {paracousti_files[fix]}",
            leave=False,
        ):
            metric_unweighted = calc_SPL_SEL_metrics(pf[pix])
            for key in metric_unweighted.keys():
                Unweighted_Dict[key][pix] = metric_unweighted[key]
            for weight in Weighted_Dict.keys():
                metric_weighted = calc_SPL_SEL_metrics(pf[pix], w_fxns[weight])
                for key in metric_weighted.keys():
                    Weighted_Dict[weight][key][pix] = metric_weighted[key]

        # Reshape to original dimensions
        for key in Unweighted_Dict.keys():
            Unweighted_Dict[key] = Unweighted_Dict[key].reshape(orig_shape[:-1])
        for weight in Weighted_Dict.keys():
            for key in metric_weighted.keys():
                Weighted_Dict[weight][key] = Weighted_Dict[weight][key].reshape(
                    orig_shape[:-1]
                )

        # Rename weighting to single key and simplify to one dictionary for saving
        Metrics_Dicts = {}
        for key in Unweighted_Dict.keys():
            Metrics_Dicts[key] = Unweighted_Dict[key]
        for weight in Weighted_Dict.keys():
            for key in metric_weighted.keys():
                Metrics_Dicts[f"{weight}_{key}"] = Weighted_Dict[weight][key]

        # Add to netcdf and save
        for key in Metrics_Dicts.keys():
            DS[key] = (
                ["Nz", "Ny", "Nx"],
                Metrics_Dicts[key],
                {"units": "dB", "description": var_descriptions[key]},
            )
        DS.to_netcdf(os.path.join(save_path, paracousti_files[fix]))

    return "Processed"
