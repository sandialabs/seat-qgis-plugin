# pylint: disable=too-many-statements
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-branches

"""
/***************************************************************************.

 shear_stress_module.py
 Copyright 2023, Integral Consulting Inc. All rights reserved.

 PURPOSE: module for calcualting shear stress (sediment mobility)
 change from a shear stress stressor

 PROJECT INFORMATION:
 Name: SEAT - Spatial and Environmental Assessment Toolkit
 Number: C1308

 AUTHORS
  Timothy Nelson (tnelson@integral-corp.com)
  Sam McWilliams (smcwilliams@integral-corp.com)
  Eben Pendelton

 NOTES (Data descriptions and any script specific notes)
	1. called by stressor_receptor_calc.py
"""

import os
from typing import Optional, Tuple, Dict
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from netCDF4 import Dataset  # pylint: disable=no-name-in-module

from seat.utils.stressor_utils import (
    estimate_grid_spacing,
    create_structured_array_from_unstructured,
    calc_receptor_array,
    trim_zeros,
    create_raster,
    numpy_array_to_raster,
    classify_layer_area,
    bin_layer,
    classify_layer_area_2nd_constraint,
    resample_structured_grid,
    secondary_constraint_geotiff_to_numpy,
)


def critical_shear_stress(
    d_meters: NDArray[np.float64],
    rhow: float = 1024,
    nu: float = 1e-6,
    s: float = 2.65,
    g: float = 9.81,
) -> NDArray[np.float64]:
    """
    Calculate critical shear stress from grain size.

    Parameters
    ----------
    d_meters : Array
        grain size in meters.
    rhow : scalar, optional
        density of water in kg/m3. The default is 1024.
    nu : scalar, optional
        kinematic viscosity of water. The default is 1e-6.
    s : scalar, optional
        specific gravity of sediment. The default is 2.65.
    g : scalar, optional
        acceleratin due to gravity. The default is 9.81.

    Returns
    -------
    taucrit : TYPE
        DESCRIPTION.

    """
    d_star = ((g * (s - 1)) / nu**2) ** (1 / 3) * d_meters
    sh_cr = (0.3 / (1 + 1.2 * d_star)) + 0.055 * (1 - np.exp(-0.02 * d_star))
    taucrit = rhow * (s - 1) * g * d_meters * sh_cr  # in Pascals
    return taucrit


def classify_mobility(
    mobility_parameter_dev: NDArray[np.float64],
    mobility_parameter_nodev: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    classifies sediment mobility from device runs to no device runs.

    Parameters
    ----------
    mobility_parameter_dev : Array
        mobility parameter (tau/tau_crit) for with device runs.
    mobility_parameter_nodev : TYPE
        mobility parameter (tau/tau_crit) for without (baseline) device runs.

    Returns
    -------
    mobility_classification : array
        Numerically classified array where,
        3 = New Erosion
        2 = Increased Erosion
        1 = Reduced Erosion
        0 = No Change
        -1 = Reduced Deposition
        -2 = Increased Deposition
        -3 = New Deposition
    """

    mobility_classification = np.zeros(mobility_parameter_dev.shape)
    # New Erosion = 3
    mobility_classification = np.where(
        (
            (mobility_parameter_nodev < mobility_parameter_dev)
            & (mobility_parameter_nodev < 1)
            & (mobility_parameter_dev >= 1)
        ),
        3,
        mobility_classification,
    )
    # Increased Erosion (Tw>Tb) & (Tw-Tb)>1 = 2
    mobility_classification = np.where(
        (
            (mobility_parameter_dev > mobility_parameter_nodev)
            & (mobility_parameter_nodev >= 1)
            & (mobility_parameter_dev >= 1)
        ),
        2,
        mobility_classification,
    )
    # Reduced Erosion (Tw<Tb) & (Tw-Tb)>1 = 1
    mobility_classification = np.where(
        (
            (mobility_parameter_dev < mobility_parameter_nodev)
            & (mobility_parameter_nodev >= 1)
            & (mobility_parameter_dev >= 1)
        ),
        1,
        mobility_classification,
    )
    # Reduced Deposition (Tw>Tb) & (Tw-Tb)<1 = -1
    mobility_classification = np.where(
        (
            (mobility_parameter_dev > mobility_parameter_nodev)
            & (mobility_parameter_nodev < 1)
            & (mobility_parameter_dev < 1)
        ),
        -1,
        mobility_classification,
    )
    # Increased Deposition (Tw>Tb) & (Tw-Tb)>1 = -2
    mobility_classification = np.where(
        (
            (mobility_parameter_dev < mobility_parameter_nodev)
            & (mobility_parameter_nodev < 1)
            & (mobility_parameter_dev < 1)
        ),
        -2,
        mobility_classification,
    )
    # New Deposition = -3
    mobility_classification = np.where(
        (
            (mobility_parameter_dev < mobility_parameter_nodev)
            & (mobility_parameter_nodev >= 1)
            & (mobility_parameter_dev < 1)
        ),
        -3,
        mobility_classification,
    )
    # NoChange = 0
    return mobility_classification


def check_grid_define_vars(dataset: Dataset) -> tuple[str, str, str, str]:
    """
    Determins the type of grid and corresponding shear stress variable name and coordiante names

    Parameters
    ----------
    dataset : netdcf (.nc) dataset
        netdcf (.nc) dataset.

    Returns
    -------
    gridtype : string
        "structured" or "unstructured".
    xvar : str
        name of x-coordinate variable.
    yvar : str
        name of y-coordiante variable.
    tauvar : str
        name of shear stress variable.

    """
    variable_names = list(dataset.variables)
    if "TAUMAX" in variable_names:
        gridtype = "structured"
        tauvar = "TAUMAX"
    else:
        gridtype = "unstructured"
        tauvar = "taus"
    xvar, yvar = dataset.variables[tauvar].coordinates.split()
    return gridtype, xvar, yvar, tauvar


def calculate_shear_stress_stressors(
    fpath_nodev: str,
    fpath_dev: str,
    probabilities_file: str,
    receptor_filename: Optional[str] = None,
    latlon: bool = True,
    value_selection: Optional[str] = None,
) -> Tuple[
    list[NDArray[np.float64]],
    NDArray[np.float64],
    NDArray[np.float64],
    float,
    float,
    str,
]:
    """
    Calculates the stressor layers as arrays from model and parameter input.

    Parameters
    ----------
    fpath_nodev : str
        Directory path to the baseline/no device model run netcdf files.
    fpath_dev : str
        Directory path to the with device model run netcdf files.
    probabilities_file : str
        File path to probabilities/bondary condition *.csv file.
    receptor_filename : str, optional
        File path to the recetptor file (*.csv or *.tif). The default is None.
    latlon : Bool, optional
        True is coordinates are lat/lon. The default is True.
    value_selection : str, optional
        Temporal selection of shears stress (not currently used). The default is 'MAX'.

    Raises
    ------
    Exception
        "Number of device runs files must be the same as no device runs files".

    Returns
    -------
    listOfFiles : list
        2D arrays of:
        [0] tau_diff
        [1] mobility_parameter_nodev
        [2] mobility_parameter_dev
        [3] mobility_parameter_diff
        [4] mobility_classification
        [5] receptor array
        [6] tau_combined_dev
        [7] tau_combined_nodev
    rx : array
        X-Coordiantes.
    ry : array
        Y-Coordinates.
    dx : scalar
        x-spacing.
    dy : scalar
        y-spacing.
    gridtype : str
        grid type [structured or unstructured].

    """
    if not os.path.exists(fpath_nodev):
        raise FileNotFoundError(f"The file {fpath_nodev} does not exist.")
    if not os.path.exists(fpath_dev):
        raise FileNotFoundError(f"The file {fpath_dev} does not exist.")

    xcor, ycor = None, None

    files_nodev = [i for i in os.listdir(fpath_nodev) if i.endswith(".nc")]
    files_dev = [i for i in os.listdir(fpath_dev) if i.endswith(".nc")]

    # Load and sort files
    if len(files_nodev) == 1 & len(files_dev) == 1:
        # asumes a concatonated files with shape
        # [run_num, time, rows, cols]

        with Dataset(
            os.path.join(fpath_dev, files_dev[0])
        ) as file_dev_present, Dataset(
            os.path.join(fpath_nodev, files_nodev[0])
        ) as file_dev_notpresent:
            gridtype, xvar, yvar, tauvar = check_grid_define_vars(file_dev_present)
            xcor = file_dev_present.variables[xvar][:].data
            ycor = file_dev_present.variables[yvar][:].data
            tau_dev = file_dev_present.variables[tauvar][:]
            tau_nodev = file_dev_notpresent.variables[tauvar][:]

    # same number of files, file name must be formatted with either run number
    elif len(files_nodev) == len(files_dev):
        # asumes each run is separate with the some_name_RunNum_map.nc,
        # where run number comes at the last underscore before _map.nc
        run_num_nodev = np.zeros((len(files_nodev)))
        for ic, file in enumerate(files_nodev):
            run_num_nodev[ic] = int(file.split(".")[0].split("_")[-2])
        run_num_dev = np.zeros((len(files_dev)))
        for ic, file in enumerate(files_dev):
            run_num_dev[ic] = int(file.split(".")[0].split("_")[-2])

        # ensure run oder for nodev matches dev files
        if np.any(run_num_nodev != run_num_dev):
            adjust_dev_order = []
            for ri in run_num_nodev:
                adjust_dev_order = np.append(
                    adjust_dev_order, np.flatnonzero(run_num_dev == ri)
                )
            files_dev = [files_dev[int(i)] for i in adjust_dev_order]
            run_num_dev = [run_num_dev[int(i)] for i in adjust_dev_order]
        df = pd.DataFrame(
            {
                "files_nodev": files_nodev,
                "run_num_nodev": run_num_nodev,
                "files_dev": files_dev,
                "run_num_dev": run_num_dev,
            }
        )
        df = df.sort_values(by="run_num_dev")
        first_run = True
        ir = 0
        for _, row in df.iterrows():
            with Dataset(
                os.path.join(fpath_nodev, row.files_nodev)
            ) as file_dev_notpresent, Dataset(
                os.path.join(fpath_dev, row.files_dev)
            ) as file_dev_present:
                gridtype, xvar, yvar, tauvar = check_grid_define_vars(file_dev_present)

                if first_run:
                    tmp = file_dev_notpresent.variables[tauvar][:].data
                    if gridtype == "structured":
                        tau_nodev = np.zeros(
                            (df.shape[0], tmp.shape[0], tmp.shape[1], tmp.shape[2])
                        )
                        tau_dev = np.zeros(
                            (df.shape[0], tmp.shape[0], tmp.shape[1], tmp.shape[2])
                        )
                    else:
                        tau_nodev = np.zeros((df.shape[0], tmp.shape[0], tmp.shape[1]))
                        tau_dev = np.zeros((df.shape[0], tmp.shape[0], tmp.shape[1]))
                    xcor = file_dev_notpresent.variables[xvar][:].data
                    ycor = file_dev_notpresent.variables[yvar][:].data
                    first_run = False
                tau_nodev[ir, :] = file_dev_notpresent.variables[tauvar][:].data
                tau_dev[ir, :] = file_dev_present.variables[tauvar][:].data

                ir += 1
    else:
        raise ValueError(
            f"Number of device runs ({len(files_dev)}) must be the same "
            f"as no device runs ({len(files_nodev)})."
        )
    # Finished loading and sorting files

    if gridtype == "structured":

        if (xcor[0, 0] == 0) & (xcor[-1, 0] == 0):
            # at least for some runs the boundary has 0 coordinates. Check and fix.
            xcor, ycor, tau_nodev, tau_dev = trim_zeros(xcor, ycor, tau_nodev, tau_dev)

    if not probabilities_file == "":
        if not os.path.exists(probabilities_file):
            raise FileNotFoundError(f"The file {probabilities_file} does not exist.")
        # Load BC file with probabilities and find appropriate probability
        bc_probability = pd.read_csv(probabilities_file, delimiter=",")
        bc_probability["run_num"] = bc_probability["run number"] - 1
        bc_probability = bc_probability.sort_values(by="run number")
        bc_probability["probability"] = bc_probability["% of yr"].values / 100
        # bc_probability
        if "Exclude" in bc_probability.columns:
            bc_probability = bc_probability[
                ~(
                    (bc_probability["Exclude"] == "x")
                    | (bc_probability["Exclude"] == "X")
                )
            ]
    else:  # assume run_num in file name is return interval
        bc_probability = pd.DataFrame()
        # ignore number and start sequentially from zero
        bc_probability["run_num"] = np.arange(0, tau_dev.shape[0])
        # assumes run_num in name is the return interval
        bc_probability["probability"] = 1 / df.run_num_dev.to_numpy()
        bc_probability["probability"] = (
            bc_probability["probability"] / bc_probability["probability"].sum()
        )  # rescale to ensure = 1

    # Calculate Stressor and Receptors
    if value_selection == "Maximum":
        tau_dev = np.nanmax(tau_dev, axis=1, keepdims=True)  # max over time
        tau_nodev = np.nanmax(tau_nodev, axis=1, keepdims=True)  # max over time
    elif value_selection == "Mean":
        tau_dev = np.nanmean(tau_dev, axis=1, keepdims=True)  # mean over time
        tau_nodev = np.nanmean(tau_nodev, axis=1, keepdims=True)  # mean over time
    elif value_selection == "Final Timestep":
        tau_dev = tau_dev[:, -2:-1, :]  # final timestep
        tau_nodev = tau_nodev[:, -2:-1, :]  # final timestep
    else:
        tau_dev = np.nanmax(tau_dev, axis=1, keepdims=True)  # default to max over time
        tau_nodev = np.nanmax(
            tau_nodev, axis=1, keepdims=True
        )  # default to max over time

    # initialize arrays
    if gridtype == "structured":
        tau_combined_nodev = np.zeros(np.shape(tau_nodev[0, 0, :, :]))
        tau_combined_dev = np.zeros(np.shape(tau_dev[0, 0, :, :]))
    else:
        tau_combined_nodev = np.zeros(np.shape(tau_nodev)[-1])
        tau_combined_dev = np.zeros(np.shape(tau_dev)[-1])

    for run_number, prob in zip(
        bc_probability["run_num"].values, bc_probability["probability"].values
    ):
        tau_combined_nodev = tau_combined_nodev + prob * tau_nodev[run_number, -1, :]
        tau_combined_dev = tau_combined_dev + prob * tau_dev[run_number, -1, :]

    receptor_array = calc_receptor_array(receptor_filename, xcor, ycor, latlon=latlon)
    taucrit = critical_shear_stress(
        d_meters=receptor_array * 1e-6, rhow=1024, nu=1e-6, s=2.65, g=9.81
    )  # units N/m2 = Pa
    tau_diff = tau_combined_dev - tau_combined_nodev
    mobility_parameter_nodev = tau_combined_nodev / taucrit
    mobility_parameter_nodev = np.where(
        receptor_array == 0, 0, mobility_parameter_nodev
    )
    mobility_parameter_dev = tau_combined_dev / taucrit
    mobility_parameter_dev = np.where(receptor_array == 0, 0, mobility_parameter_dev)
    # Calculate risk metrics over all runs

    mobility_parameter_diff = mobility_parameter_dev - mobility_parameter_nodev

    # EQ 7 in Jones et al. (2018) doi:10.3390/en11082036
    risk = np.round(
        mobility_parameter_dev
        * (
            (mobility_parameter_dev - mobility_parameter_nodev)
            / np.abs(mobility_parameter_dev - mobility_parameter_nodev)
        )
    ) + (mobility_parameter_dev - mobility_parameter_nodev)

    if gridtype == "structured":
        mobility_classification = classify_mobility(
            mobility_parameter_dev, mobility_parameter_nodev
        )
        dx = np.nanmean(np.diff(xcor[:, 0]))
        dy = np.nanmean(np.diff(ycor[0, :]))
        rx = xcor
        ry = ycor
        dict_of_arrays = {
            "shear_stress_without_devices": tau_combined_nodev,
            "shear_stress_with_devices": tau_combined_dev,
            "shear_stress_difference": tau_diff,
            "sediment_mobility_without_devices": mobility_parameter_nodev,
            "sediment_mobility_with_devices": mobility_parameter_dev,
            "sediment_mobility_difference": mobility_parameter_diff,
            "sediment_mobility_classified": mobility_classification,
            "sediment_grain_size": receptor_array,
            "shear_stress_risk_metric": risk,
        }
    else:  # unstructured
        dxdy = estimate_grid_spacing(xcor, ycor, nsamples=100)
        dx = dxdy
        dy = dxdy
        rx, ry, tau_diff_struct = create_structured_array_from_unstructured(
            xcor, ycor, tau_diff, dxdy, flatness=0.2
        )
        _, _, tau_combined_dev_struct = create_structured_array_from_unstructured(
            xcor, ycor, tau_combined_dev, dxdy, flatness=0.2
        )
        _, _, tau_combined_nodev_struct = create_structured_array_from_unstructured(
            xcor, ycor, tau_combined_nodev, dxdy, flatness=0.2
        )
        if not ((receptor_filename is None) or (receptor_filename == "")):
            _, _, mobility_parameter_nodev_struct = (
                create_structured_array_from_unstructured(
                    xcor, ycor, mobility_parameter_nodev, dxdy, flatness=0.2
                )
            )
            _, _, mobility_parameter_dev_struct = (
                create_structured_array_from_unstructured(
                    xcor, ycor, mobility_parameter_dev, dxdy, flatness=0.2
                )
            )
            _, _, mobility_parameter_diff_struct = (
                create_structured_array_from_unstructured(
                    xcor, ycor, mobility_parameter_diff, dxdy, flatness=0.2
                )
            )
            _, _, receptor_array_struct = create_structured_array_from_unstructured(
                xcor, ycor, receptor_array, dxdy, flatness=0.2
            )
            _, _, risk_struct = create_structured_array_from_unstructured(
                xcor, ycor, risk, dxdy, flatness=0.2
            )
        else:
            mobility_parameter_nodev_struct = np.nan * tau_diff_struct
            mobility_parameter_dev_struct = np.nan * tau_diff_struct
            mobility_parameter_diff_struct = np.nan * tau_diff_struct
            receptor_array_struct = np.nan * tau_diff_struct
            risk_struct = np.nan * tau_diff_struct
        mobility_classification = classify_mobility(
            mobility_parameter_dev_struct, mobility_parameter_nodev_struct
        )
        mobility_classification = np.where(
            np.isnan(tau_diff_struct), -100, mobility_classification
        )

        dict_of_arrays = {
            "shear_stress_without_devices": tau_combined_nodev_struct,
            "shear_stress_with_devices": tau_combined_dev_struct,
            "shear_stress_difference": tau_diff_struct,
            "sediment_mobility_without_devices": mobility_parameter_nodev_struct,
            "sediment_mobility_with_devices": mobility_parameter_dev_struct,
            "sediment_mobility_difference": mobility_parameter_diff_struct,
            "sediment_mobility_classified": mobility_classification,
            "sediment_grain_size": receptor_array_struct,
            "shear_stress_risk_metric": risk_struct,
        }

    return dict_of_arrays, rx, ry, dx, dy, gridtype


def run_shear_stress_stressor(
    dev_present_file: str,
    dev_notpresent_file: str,
    probabilities_file: str,
    crs: int,
    output_path: str,
    receptor_filename: Optional[str] = None,
    secondary_constraint_filename: Optional[str] = None,
    value_selection: Optional[str] = None,
) -> Dict[str, str]:
    """
    creates geotiffs and area change statistics files for shear stress change

    Parameters
    ----------
    dev_present_file : str
        Directory path to the baseline/no device model run netcdf files.
    dev_notpresent_file : str
        Directory path to the baseline/no device model run netcdf files.
    probabilities_file : str
        File path to probabilities/bondary condition *.csv file.
    crs : scalar
        Coordiante Reference System / EPSG code.
    output_path : str
        File directory to save output.
    receptor_filename : str, optional
        File path to the recetptor file (*.csv or *.tif). The default is None.
    secondary_constraint_filename: str, optional
        File path to the secondary constraint file (*.tif). The default is None.

    Returns
    -------
    output_rasters : dict
        key = names of output rasters, val = full path to raster:
    """

    os.makedirs(
        output_path, exist_ok=True
    )  # create output directory if it doesn't exist

    dict_of_arrays, rx, ry, dx, dy, gridtype = calculate_shear_stress_stressors(
        fpath_nodev=dev_notpresent_file,
        fpath_dev=dev_present_file,
        probabilities_file=probabilities_file,
        receptor_filename=receptor_filename,
        latlon=crs == 4326,
        value_selection=value_selection,
    )

    if not ((receptor_filename is None) or (receptor_filename == "")):
        use_numpy_arrays = [
            "shear_stress_without_devices",
            "shear_stress_with_devices",
            "shear_stress_difference",
            "sediment_mobility_without_devices",
            "sediment_mobility_with_devices",
            "sediment_mobility_difference",
            "sediment_mobility_classified",
            "sediment_grain_size",
            "shear_stress_risk_metric",
        ]
    else:
        use_numpy_arrays = [
            "shear_stress_without_devices",
            "shear_stress_with_devices",
            "shear_stress_difference",
        ]

    if not (
        (secondary_constraint_filename is None) or (secondary_constraint_filename == "")
    ):
        if not os.path.exists(secondary_constraint_filename):
            raise FileNotFoundError(
                f"The file {secondary_constraint_filename} does not exist."
            )
        rrx, rry, constraint = secondary_constraint_geotiff_to_numpy(
            secondary_constraint_filename
        )
        dict_of_arrays["shear_stress_risk_layer"] = resample_structured_grid(
            rrx, rry, constraint, rx, ry, interpmethod="nearest"
        )
        use_numpy_arrays.append("shear_stress_risk_layer")

    numpy_array_names = [i + ".tif" for i in use_numpy_arrays]

    output_rasters = []
    for array_name, use_numpy_array in zip(numpy_array_names, use_numpy_arrays):
        if gridtype == "structured":
            numpy_array = np.flip(np.transpose(dict_of_arrays[use_numpy_array]), axis=0)
        else:
            numpy_array = np.flip(dict_of_arrays[use_numpy_array], axis=0)

        cell_resolution = [dx, dy]
        if crs == 4326:
            rxx = np.where(rx > 180, rx - 360, rx)
            bounds = [rxx.min() - dx / 2, ry.max() - dy / 2]
        else:
            bounds = [rx.min() - dx / 2, ry.max() - dy / 2]
        rows, cols = numpy_array.shape
        # create an ouput raster given the stressor file path
        output_rasters.append(os.path.join(output_path, array_name))
        output_raster = create_raster(
            os.path.join(output_path, array_name), cols, rows, nbands=1
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
        output_raster = None

    # Area calculations pull form rasters to ensure uniformity
    bin_layer(
        os.path.join(output_path, "shear_stress_difference.tif"),
        receptor_filename=None,
        receptor_names=None,
        latlon=crs == 4326,
    ).to_csv(os.path.join(output_path, "shear_stress_difference.csv"), index=False)
    if not (
        (secondary_constraint_filename is None) or (secondary_constraint_filename == "")
    ):
        bin_layer(
            os.path.join(output_path, "shear_stress_difference.tif"),
            receptor_filename=os.path.join(output_path, "shear_stress_risk_layer.tif"),
            receptor_names=None,
            limit_receptor_range=[0, np.inf],
            latlon=crs == 4326,
            receptor_type="risk layer",
        ).to_csv(
            os.path.join(
                output_path, "shear_stress_difference_at_secondary_constraint.csv"
            ),
            index=False,
        )
    if not ((receptor_filename is None) or (receptor_filename == "")):
        bin_layer(
            os.path.join(output_path, "shear_stress_difference.tif"),
            receptor_filename=os.path.join(output_path, "sediment_grain_size.tif"),
            receptor_names=None,
            limit_receptor_range=[0, np.inf],
            latlon=crs == 4326,
            receptor_type="grain size",
        ).to_csv(
            os.path.join(
                output_path, "shear_stress_difference_at_sediment_grain_size.csv"
            ),
            index=False,
        )

        bin_layer(
            os.path.join(output_path, "sediment_mobility_difference.tif"),
            receptor_filename=None,
            receptor_names=None,
            limit_receptor_range=[0, np.inf],
            latlon=crs == 4326,
        ).to_csv(
            os.path.join(output_path, "sediment_mobility_difference.csv"), index=False
        )

        bin_layer(
            os.path.join(output_path, "sediment_mobility_difference.tif"),
            receptor_filename=os.path.join(output_path, "sediment_grain_size.tif"),
            receptor_names=None,
            limit_receptor_range=[0, np.inf],
            latlon=crs == 4326,
            receptor_type="grain size",
        ).to_csv(
            os.path.join(
                output_path, "sediment_mobility_difference_at_sediment_grain_size.csv"
            ),
            index=False,
        )

        bin_layer(
            os.path.join(output_path, "shear_stress_risk_metric.tif"),
            receptor_filename=None,
            receptor_names=None,
            limit_receptor_range=[0, np.inf],
            latlon=crs == 4326,
        ).to_csv(os.path.join(output_path, "shear_stress_risk_metric.csv"), index=False)

        bin_layer(
            os.path.join(output_path, "shear_stress_risk_metric.tif"),
            receptor_filename=os.path.join(output_path, "sediment_grain_size.tif"),
            receptor_names=None,
            limit_receptor_range=[0, np.inf],
            latlon=crs == 4326,
            receptor_type="grain size",
        ).to_csv(
            os.path.join(
                output_path, "shear_stress_risk_metric_at_sediment_grain_size.csv"
            ),
            index=False,
        )

        classify_layer_area(
            os.path.join(output_path, "sediment_mobility_classified.tif"),
            at_values=[-3, -2, -1, 0, 1, 2, 3],
            value_names=[
                "New Deposition",
                "Increased Deposition",
                "Reduced Deposition",
                "No Change",
                "Reduced Erosion",
                "Increased Erosion",
                "New Erosion",
            ],
            latlon=crs == 4326,
        ).to_csv(
            os.path.join(output_path, "sediment_mobility_classified.csv"), index=False
        )

        classify_layer_area(
            os.path.join(output_path, "sediment_mobility_classified.tif"),
            receptor_filename=os.path.join(output_path, "sediment_grain_size.tif"),
            at_values=[-3, -2, -1, 0, 1, 2, 3],
            value_names=[
                "New Deposition",
                "Increased Deposition",
                "Reduced Deposition",
                "No Change",
                "Reduced Erosion",
                "Increased Erosion",
                "New Erosion",
            ],
            limit_receptor_range=[0, np.inf],
            latlon=crs == 4326,
            receptor_type="grain size",
        ).to_csv(
            os.path.join(
                output_path, "sediment_mobility_classified_at_sediment_grain_size.csv"
            ),
            index=False,
        )

        if not (
            (secondary_constraint_filename is None)
            or (secondary_constraint_filename == "")
        ):
            bin_layer(
                os.path.join(output_path, "sediment_mobility_difference.tif"),
                receptor_filename=os.path.join(
                    output_path, "shear_stress_risk_layer.tif"
                ),
                receptor_names=None,
                limit_receptor_range=[0, np.inf],
                latlon=crs == 4326,
                receptor_type="risk layer",
            ).to_csv(
                os.path.join(
                    output_path,
                    "sediment_mobility_difference_at_shear_stress_risk_layer.csv",
                ),
                index=False,
            )

            bin_layer(
                os.path.join(output_path, "shear_stress_risk_metric.tif"),
                receptor_filename=os.path.join(
                    output_path, "shear_stress_risk_layer.tif"
                ),
                receptor_names=None,
                limit_receptor_range=[0, np.inf],
                latlon=crs == 4326,
                receptor_type="risk layer",
            ).to_csv(
                os.path.join(
                    output_path,
                    "shear_stress_risk_metric_at_shear_stress_risk_layer.csv",
                ),
                index=False,
            )

            classify_layer_area_2nd_constraint(
                raster_to_sample=os.path.join(
                    output_path, "sediment_mobility_difference.tif"
                ),
                secondary_constraint_filename=os.path.join(
                    output_path, "shear_stress_risk_layer.tif"
                ),
                at_raster_values=[-3, -2, -1, 0, 1, 2, 3],
                at_raster_value_names=[
                    "New Deposition",
                    "Increased Deposition",
                    "Reduced Deposition",
                    "No Change",
                    "Reduced Erosion",
                    "Increased Erosion",
                    "New Erosion",
                ],
                limit_constraint_range=[0, np.inf],
                latlon=crs == 4326,
                receptor_type="risk layer",
            ).to_csv(
                os.path.join(
                    output_path,
                    "sediment_mobility_difference_at_shear_stress_risk_layer.csv",
                ),
                index=False,
            )
    output = {}
    for val in output_rasters:
        output[os.path.basename(os.path.normpath(val)).split(".")[0]] = val
    return output
