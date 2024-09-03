# power_module.py

# pylint: disable=too-many-statements
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-branches


"""
This module provides functionalities related to power calculations.

AUTHORS:
    - Timothy R. Nelson (TRN) - tnelson@integral-corp.com

NOTES:
    1. .OUT files and .pol files must be in the same folder.
    2. .OUT file is format sensitive.

CHANGE HISTORY:
    - 2022-08-01: TRN - File created.
    - 2023-08-05: TRN - Comments and slight edits.
"""


import io
import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FormatStrFormatter

# Obstacle Polygon and Device Positions
def read_obstacle_polygon_file(power_device_configuration_file):
    """
    reads the obstacle polygon file

    Parameters
    ----------
    power_device_configuration_file : str
        filepath of .pol file.

    Returns
    -------
    obstacles : Dict
        xy of each obstacle.

    """
    try:
        with io.open(power_device_configuration_file, "r", encoding="utf-8") as inf:
            lines = inf.readlines()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"File not found: {power_device_configuration_file}") from exc

    ic = 0
    obstacles = {}
    while ic < len(lines) - 1:
        if 'Obstacle' in lines[ic]:
            obstacle = lines[ic].strip()
            obstacles[obstacle] = {}
            ic += 1  # skip to next line
            nrows = int(lines[ic].split()[0])
            ic += 1  # skip to next line
            x = []
            y = []
            for _ in range(nrows):  # read polygon
                xi, yi = [float(i) for i in lines[ic].split()]
                x = np.append(x, xi)
                y = np.append(y, yi)
                ic += 1  # skip to next line
            obstacles[obstacle] = np.vstack((x, y)).T
        else:
            ic += 1
    return obstacles


def find_mean_point_of_obstacle_polygon(obstacles):
    """
    Calculates the center of each obstacle.

    Parameters
    ----------
    obstacles : Dict
        x,y of each obstacle.

    Returns
    -------
    centroids : array
        Centroid of each obstacle.

    """
    centroids = np.empty((0, 3), dtype=int)
    for ic, obstacle in enumerate(obstacles.keys()):
        centroids = np.vstack(
            (
                centroids,
                [
                    ic,
                    np.nanmean(obstacles[obstacle][:, 0]),
                    np.nanmean(obstacles[obstacle][:, 1]),
                ],
            )
        )
    return centroids


def plot_test_obstacle_locations(obstacles):
    """
    Creates a plot of the spatial distribution and location of each obstacle.

    Parameters
    ----------
    obstacles : Dict
        xy of each obstacle.

    Returns
    -------
    fig : pyplot figure handle
        pyplot figure handle.

    """
    fig, ax = plt.subplots(figsize=(10, 10))
    for obstacle in obstacles.keys():
        ax.plot(
            obstacles[obstacle][:, 0],
            obstacles[obstacle][:, 1],
            ".",
            markersize=3,
            alpha=0,
        )
        ax.text(
            obstacles[obstacle][0, 0],
            obstacles[obstacle][0, 1],
            f"{obstacle}",
            fontsize=8,
        )
        ax.text(
            obstacles[obstacle][1, 0],
            obstacles[obstacle][1, 1],
            f"{obstacle}",
            fontsize=8,
        )
    fig.tight_layout()
    return fig


def centroid_diffs(centroids, centroid):
    """
    Determines the closest centroid pair

    Parameters
    ----------
    centroids : Dict
        dimensions M,N with each M [index, x , y]
    centroid : array
        single x,y.

    Returns
    -------
    pair : list
        index of closest centroid.

    """

    diff = centroids[:, 1:] - centroid[1:]
    min_arg = np.nanargmin(np.abs(diff[:, -1] - diff[:, 0]))
    pair = [int(centroid[0]), int(centroids[min_arg, 0])]
    return pair


def extract_device_location(obstacles, device_index):
    """
    Creates a dictionary summary of each device location

    Parameters
    ----------
    obstacles : TYPE
        DESCRIPTION.
    device_index : TYPE
        DESCRIPTION.

    Returns
    -------
    devices_df : TYPE
        DESCRIPTION.

    """
    devices = {}
    for device, [ix1, ix2] in enumerate(device_index):
        key = f"{device+1:03.0f}"
        devices[key] = {}
        xy = obstacles[f"Obstacle {ix1+1}"]
        xy = np.vstack((xy, obstacles[f"Obstacle {ix2+1}"]))
        # create polygon from bottom left to upper right assuming rectangular
        x = xy[:, 0]
        y = xy[:, 1]
        devices[key]["polyx"] = [np.nanmin(x), np.nanmin(x), np.nanmax(x), np.nanmax(x)]
        devices[key]["polyy"] = [np.nanmin(y), np.nanmax(y), np.nanmax(y), np.nanmin(y)]
        devices[key]["lower_left"] = [np.nanmin(x), np.nanmin(y)]
        devices[key]["centroid"] = [np.nanmean(x), np.nanmean(y)]
        devices[key]["width"] = np.nanmax(x) - np.nanmin(x)
        devices[key]["height"] = np.nanmax(y) - np.nanmin(y)
    devices_df = pd.DataFrame.from_dict(devices, orient="index")
    return devices_df


def pair_devices(centroids):
    """
    Determins the two intersecting obstacles to that create a device.

    Parameters
    ----------
    centroids : TYPE
        DESCRIPTION.

    Returns
    -------
    devices : TYPE
        DESCRIPTION.

    """
    devices = np.empty((0, 2), dtype=int)
    while len(centroids) > 0:
        # print(centroids)
        # must have dimensions M,N with each M [index, x , y]
        pair = centroid_diffs(centroids[1:, :], centroids[0, :])
        devices = np.vstack((devices, pair))
        centroids = centroids[~np.isin(centroids[:, 0], pair), :]
    return devices

# # use https://stackoverflow.com/questions/10550477/how-do-i-set-color-to-rectangle-in-matplotlib
# # to create rectangles from the polygons above
# # scale color based on power device power range from 0 to max of array
# # This way the plot is array and grid independent, only based on centroid and device size,
# could make size variable if necessary.
#


def create_power_heatmap(device_power, crs=None):
    """
    Creates a heatmap of device location and power as cvalue.

    Parameters
    ----------
    device_power : dataframe
        device_power dataframe.
    crs : int
        Coordinate Reverence Systems EPSG number

    Returns
    -------
    fig : matplotlib figure handle
        matplotlib figure handle.

    """
    adjust_x = -360 if crs == 4326 else 0

    fig, ax = plt.subplots(figsize=(6, 4))
    lowerx = np.inf
    lowery = np.inf
    upperx = -np.inf
    uppery = -np.inf
    # cmap = ListedColormap(plt.get_cmap('Greens')(np.linspace(0.1, 1, 256)))
    # # skip too light colors
    cmap = ListedColormap(
        plt.get_cmap("turbo")(np.linspace(0.1, 1, 256))
    )  # skip too light colors
    # norm = plt.Normalize(device_power['Power [W]'].min(), device_power['Power [W]'].max())
    norm = plt.Normalize(
        0.9 * device_power["Power [W]"].min() * 1e-6,
        device_power["Power [W]"].max() * 1e-6,
    )
    for _, device in device_power.iterrows():
        # print(device)
        ax.add_patch(
            Rectangle((device.lower_left[0]+adjust_x, device.lower_left[1]),
                      np.nanmax([device.width, device.height]),
                      np.nanmax([device.width, device.height]),
                      color=cmap(norm(device['Power [W]']*1e-6))))
        lowerx = np.nanmin([lowerx, device.lower_left[0]+adjust_x])
        lowery = np.nanmin([lowery, device.lower_left[1]])
        upperx = np.nanmax(
            [upperx, device.lower_left[0]+adjust_x + device.width])
        uppery = np.nanmax([uppery, device.lower_left[1] + device.height])
    xr = np.abs(np.max([lowerx, upperx]) - np.min([lowerx, upperx]))
    yr = np.abs(np.max([lowery, uppery]) - np.min([lowery, uppery]))
    ax.set_xlim([lowerx-.05*xr, upperx+.05*xr])
    ax.set_ylim([lowery-.05*yr, uppery+.05*yr])

    cb = plt.colorbar(ScalarMappable(cmap=cmap, norm=norm), ax=ax)
    cb.set_label('MW')
    ax.ticklabel_format(useOffset=False, style='plain')
    ax.set_xlabel('Longitude [deg]')
    ax.set_ylabel('Latitude [deg]')
    ax.set_xticks(np.linspace(lowerx, upperx, 5))
    ax.set_xticklabels(ax.get_xticklabels(), ha="right", rotation=45)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%0.4f'))
    fig.tight_layout()
    return fig
# %%


def read_power_file(datafile):
    """
    Read power file and extract final set of converged data

    Parameters
    ----------
    datafile : file path
        path and file name of power file.

    Returns
    -------
    Power : 1D Numpy Array [m]
        Individual data files for each observation [m].
    total_power : Scalar
        Total power from all observations.

    """
    with io.open(datafile, "r", encoding="utf-8") as inf:
        # = io.open(datafile, "r")  # Read datafile
        for line in inf:  # iterate through each line
            if re.match("Iteration:", line):
                power_array = []
                # If a new iteration is found, initalize varialbe or overwrite existing iteration
            else:  # data
                # extract float variable from line
                power = float(line.split("=")[-1].split("W")[0].strip())
                power_array = np.append(
                    power_array, power
                )  # append data for each observation
    total_power = np.nansum(power_array)  # Total power from all observations
    return power_array, total_power


def sort_data_files_by_runnumber(bc_data, datafiles):
    """
    Sorts the data files based on the run number specified in `bc_data`.

    Parameters
    ----------
    bc_data : pd.DataFrame
        DataFrame containing the 'run number' and other metadata.
    datafiles : list
        List of data file paths.

    Returns
    -------
    List[str]
        List of sorted data file paths based on the run number.
    """
    bc_data_sorted = sort_bc_data_by_runnumber(bc_data.copy())
    return [datafiles[i] for i in bc_data_sorted.original_order.to_numpy()]


def sort_bc_data_by_runnumber(bc_data):
    """
    Sorts the `bc_data` DataFrame by the 'run number' column.

    Parameters
    ----------
    bc_data : pd.DataFrame
        DataFrame containing the 'run number' and other metadata.

    Returns
    -------
    pd.DataFrame
        Sorted DataFrame with an added 'original_order' column to track original indices.
    """
    bc_data["original_order"] = range(0, len(bc_data))
    return bc_data.sort_values(by="run number")


def reset_bc_data_order(bc_data):
    """
    Resets the order of `bc_data` DataFrame to its original order if 'original_order' column exists.

    Parameters
    ----------
    bc_data : pd.DataFrame
        DataFrame containing the 'run number' and other metadata.

    Returns
    -------
    pd.DataFrame or None
        Sorted DataFrame if 'original_order' column exists, otherwise None.
    """
    if np.isin("original_order", bc_data.columns):
        bc_data = bc_data.sort()
    return bc_data


def roundup(x, val=2):
    """
    Rounds up the number `x` to the nearest multiple of `val`.

    Parameters
    ----------
    x : float
        The number to round up.
    val : int, optional
        The value to round to the nearest multiple of (default is 2).

    Returns
    -------
    float
        The rounded-up number.
    """
    return np.ceil(x / val) * val

def calculate_power(power_files, probabilities_file, save_path=None, crs=None):
    """
    Reads the power files and calculates the total annual power based on
    hydrodynamic probabilities in probabilities_file.
    Data are saved as a csv files.
    Three files are output:
        1) Total Power among all devices for each
        hydrodynamic conditions BC_probability_Annual_SETS_wPower.csv
        2) Power per device per hydordynamic scenario. Power_per_device_per_scenario.csv
        3) Total power per device during a year, scaled by $ of year in probabilities_file

    Parameters
    ----------
    fpath : file path
        Path to bc_file and power output files.
    probabilities_file : file name
        probabilities file name with extension.
    save_path: file path
        save directory

    Returns
    -------
    devices : Dataframe
        Scaled power per device per condition.
    devices_total : Dataframe
        Total annual power per device.

    """

    if not os.path.exists(power_files):
        raise FileNotFoundError(f"The directory {power_files} does not exist.")
    if not os.path.exists(probabilities_file):
        raise FileNotFoundError(f"The file {probabilities_file} does not exist.")

    datafiles_o = [s for s in os.listdir(power_files) if s.endswith('.OUT')]
    bc_data = pd.read_csv(probabilities_file)

    datafiles = sort_data_files_by_runnumber(bc_data, datafiles_o)

    assert save_path is not None, "Specify an output directory"
    os.makedirs(save_path, exist_ok=True)

    total_power = []
    ic = 0
    for datafile in datafiles:
        p, tp = read_power_file(os.path.join(power_files, datafile))
        # print(p)
        if ic == 0:
            power_array = np.empty((len(p), 0), float)
        power_array = np.append(power_array, p[:, np.newaxis], axis=1)
        total_power = np.append(total_power, tp)
        ic += 1

    power_scaled = bc_data["% of yr"].to_numpy() * power_array
    total_power_scaled = bc_data["% of yr"] * total_power

    # Summary of power given percent of year for each array
    # need to reorder total_power and Power to run roder in
    bc_data["Power_Run_Name"] = datafiles
    # bc_data['% of yr'] * total_power
    bc_data["Power [W]"] = total_power_scaled
    bc_data.to_csv(os.path.join(save_path, "BC_probability_wPower.csv"), index=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(
        np.arange(np.shape(total_power_scaled)[0]) + 1,
        np.log10(total_power_scaled),
        width=1,
        edgecolor="black",
    )
    ax.set_xlabel("Run Scenario")
    ax.set_ylabel("Power [$log_{10}(Watts)$]")
    ax.set_title("Total Power Annual")
    fig.tight_layout()
    fig.savefig(os.path.join(
        save_path, 'Total_Scaled_Power_Bars_per_Run.png'))

    subplot_grid_size = np.sqrt(np.shape(power_scaled)[1])
    fig, axes_grid = plt.subplots(
        np.round(subplot_grid_size).astype(int),
        np.ceil(subplot_grid_size).astype(int),
        sharex=True,
        sharey=True,
        figsize=(12, 10),
    )
    nr, nc = axes_grid.shape
    axes_grid = axes_grid.flatten()
    mxy = roundup(np.log10(power_scaled.max().max()))
    ndx = np.ceil(power_scaled.shape[0] / 6)
    for ic in range(power_scaled.shape[1]):
        # fig,ax = plt.subplots()
        axes_grid[ic].bar(
            np.arange(np.shape(power_scaled)[0]) + 1,
            np.log10(power_scaled[:, ic]),
            width=1,
            edgecolor="black",
        )
        # axes_grid[ic].text(power_scaled.shape[0]/2, mxy-1,
        # f'{datafiles[ic]}', fontsize=8, ha='center', va='top')
        axes_grid[ic].set_title(f"{datafiles[ic]}", fontsize=8)
        axes_grid[ic].set_ylim([0, mxy])
        axes_grid[ic].set_xticks(np.arange(0, power_scaled.shape[0] + ndx, ndx))
        axes_grid[ic].set_xlim([0, power_scaled.shape[0] + 1])
    axes_grid = axes_grid.reshape(nr, nc)
    for ax in axes_grid[:, 0]:
        ax.set_ylabel("Power [$log_{10}(Watts)$]")
    for ax in axes_grid[-1, :]:
        ax.set_xlabel("Obstacle")
    fig.tight_layout()
    fig.savefig(os.path.join(
        save_path, 'Scaled_Power_Bars_per_run_obstacle.png'))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(
        np.arange(np.shape(power_scaled)[0]) + 1,
        np.log10(np.sum(power_scaled, axis=1)),
        width=1,
        edgecolor="black",
    )
    ax.set_xlabel("Obstacle")
    ax.set_ylabel("Power [$log_{10}(Watts)$]")
    ax.set_title("Total Obstacle Power for all Runs")
    fig.tight_layout()
    fig.savefig(os.path.join(
        save_path, 'Total_Scaled_Power_Bars_per_obstacle.png'))

    power_device_configuration_file = [s for s in os.listdir(power_files) if (
        s.endswith('.pol') | s.endswith('.Pol') | s.endswith('.POL'))]
    if len(power_device_configuration_file) > 0:

        assert len(
            power_device_configuration_file) == 1, "More than 1 *.pol file found"

        # Group arrays to devices and calculate power
        # proportionally for each scenario (datafile),
        # such that the sum of each scenario for each
        # device is the yearly totoal power for that device
        obstacles = read_obstacle_polygon_file(
            os.path.join(power_files, power_device_configuration_file[0])
        )
        fig = plot_test_obstacle_locations(obstacles)
        fig.savefig(os.path.join(save_path, "Obstacle_Locations.png"))

        centroids = find_mean_point_of_obstacle_polygon(obstacles)
        centroids_df = pd.DataFrame(data=centroids, columns=["obstacle", "X", "Y"])
        centroids_df["obstacle"] = centroids_df["obstacle"].astype(int)
        centroids_df = centroids_df.set_index(["obstacle"])
        device_index = pair_devices(centroids)
        device_index_df = pd.DataFrame(
            {
                "Device_Number": range(device_index.shape[0]),
                "Index 1": device_index[:, 0],
                "Index 2": device_index[:, 1],
                "X": centroids_df.loc[device_index[:, 0], "X"],
                "Y": centroids_df.loc[device_index[:, 0], "Y"],
            }
        )
        device_index_df["Device_Number"] = device_index_df["Device_Number"] + 1
        device_index_df = device_index_df.set_index("Device_Number")
        device_index_df.to_csv(os.path.join(save_path, "Obstacle_Matching.csv"))

        fig, ax = plt.subplots(figsize=(10, 10))
        for device in device_index_df.index.values:
            ax.plot(
                device_index_df.loc[device, "X"],
                device_index_df.loc[device, "Y"],
                ".",
                alpha=0,
            )
            ax.text(
                device_index_df.loc[device, "X"],
                device_index_df.loc[device, "Y"],
                device,
                fontsize=8,
            )
        fig.savefig(os.path.join(save_path, "Device Number Location.png"))

        device_power = np.empty((0, np.shape(power_array)[1]), dtype=float)
        for ic0, ic1 in device_index:
            device_power = np.vstack(
                (device_power, power_array[ic0, :] + power_array[ic1, :])
            )

        # device_power = power_array[0::2, :] + power_array[1::2, :]

        devices = pd.DataFrame({})
        device_power_year = device_power * bc_data["% of yr"].to_numpy()
        for ic, name in enumerate(datafiles):
            devices[name] = device_power_year[:, ic]
        devices["Device"] = np.arange(1, len(devices) + 1)
        devices = devices.set_index("Device")
        devices.to_csv(os.path.join(save_path, "Power_per_device_per_scenario.csv"))

        subplot_grid_size = np.sqrt(devices.shape[1])
        fig, axes_grid = plt.subplots(
            np.round(subplot_grid_size).astype(int),
            np.ceil(subplot_grid_size).astype(int),
            sharex=True,
            sharey=True,
            figsize=(12, 10),
        )
        nr, nc = axes_grid.shape
        axes_grid = axes_grid.flatten()
        mxy = roundup(np.log10(devices.max().max()))
        ndx = np.ceil(devices.shape[0] / 6)
        for ic, col in enumerate(devices.columns):
            # fig,ax = plt.subplots()
            axes_grid[ic].bar(
                np.arange(np.shape(devices[col])[0]) + 1,
                np.log10(devices[col].to_numpy()),
                width=1.0,
                edgecolor="black",
            )
            # axes_grid[ic].text(devices.shape[0]/2, mxy-1, f'{col}',
            # fontsize=8, ha='center', va='top')
            axes_grid[ic].set_title(f"{col}", fontsize=8)
            axes_grid[ic].set_ylim([0, mxy])
            axes_grid[ic].set_xticks(np.arange(0, devices.shape[0] + ndx, ndx))
            axes_grid[ic].set_xlim([0, devices.shape[0] + 1])
        axes_grid = axes_grid.reshape(nr, nc)
        for ax in axes_grid[:, 0]:
            ax.set_ylabel("Power [$log_{10}(Watts)$]")
        for ax in axes_grid[-1, :]:
            ax.set_xlabel("Device")
        axes_grid = axes_grid.flatten()
        fig.tight_layout()
        fig.savefig(os.path.join(
            save_path, 'Scaled_Power_per_device_per_scenario.png'))

        # power per scenario per device

        # Sum power for the entire years (all datafiles) for each device
        devices_total = pd.DataFrame({})
        devices_total["Power [W]"] = device_power_year.sum(axis=1)
        devices_total["Device"] = np.arange(1, len(devices_total) + 1)
        devices_total = devices_total.set_index("Device")
        devices_total.to_csv(os.path.join(save_path, "Power_per_device_annual.csv"))

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.bar(
            devices_total.index,
            np.log10(devices_total["Power [W]"]),
            width=1,
            edgecolor="black",
        )
        ax.set_ylabel("Power [$log_{10}(Watts)$]")
        ax.set_xlabel("Device")
        fig.savefig(os.path.join(save_path, "Total_Scaled_Power_per_Device_.png"))

        device_power = extract_device_location(obstacles, device_index)
        device_power["Power [W]"] = devices_total["Power [W]"].values
        fig = create_power_heatmap(device_power, crs=crs)
        fig.savefig(os.path.join(save_path, "Device_Power.png"), dpi=150)
        # plt.close(fig)
