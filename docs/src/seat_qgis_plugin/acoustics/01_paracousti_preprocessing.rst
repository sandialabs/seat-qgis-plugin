.. _01_paracousti_preprocessing:

Paracousti Pre-Processing 
-------------------------
Several metrics can be calculated from ParAcousti output that are useful in determining impacts to marine species. These include the weighted and unweighted sound pressure level (SPL) and sound exposure level (SEL). Due to the resource and time-intensive calculations, a stand-alone python routine is included within the SEAT package. This routine calculates the user selected metrics and weights from the existing Paracousti metrics. The calculated metrics are saved to a netcdf variable with the same dimensions as the original Paracousti netcdf.
The new netcdf with both the unweighted and weighted values are stored in a new directory. 
The weighting is detailed in `Criteria and Thresholds for US Navy Acoustic and Explosive Effects <https://nwtteis.com/portals/nwtteis/files/technical_reports/Criteria_and_Thresholds_for_U.S._Navy_Acoustic_and_Explosive_Effects_Analysis_June2017.pdf>`_.

**Underwater Weights**

    - **LWC**: Low-Frequency Cetaceans
    - **MFC**: Mid-Frequency Cetaceans
    - **HFC**: High-Frequency Cetaceans
    - **PPW**: Phocid Pinnipeds in water (earless seals)
    - **OPW**: Otariid Pinnipeds in water (eared seals)
    - **TU**: Sea Turtles
    - **SI**: Sirenia (Manatee)

**Metrics (unweighted and weighted)**:
    - **SPL_pk**: Peak Sound Pressure Level
    - **SPL_rms**: Root-mean square sound pressure level
    - **SEL**: sound exposure level (calculated as 1 second in duration)


For weighted metrics, the initial prefix is applied to the variable name with a suffix of `_weighted`

For unweighted metrics, a suffix of _flat is applied to the variable.

To run the Paracousti pre-processing routine, make sure to have a python environment activated. The python environment will need the following packages:
 

.. code-block::

    colorama==0.4.6
    contourpy==1.3.1
    cycler==0.12.1
    fonttools==4.54.1
    kiwisolver==1.4.7
    matplotlib==3.9.2
    numpy==2.1.3
    packaging==24.2
    pandas==2.2.3
    pillow==11.0.0
    pyparsing==3.2.0
    python-dateutil==2.9.0.post0
    pytz==2024.2
    scipy==1.14.1
    six==1.16.0
    tqdm==4.67.0
    tzdata==2024.2
    xarray==2024.10.0

Then, run the following code, housed in a python script:

.. code-block:: python

    from utils import paracousti_fxns 
    paracousti_directory  = r'./Path/to/Paracousti/netCDF/' 
    save_directory = r'./Path/to/save/updated/netCDF/'
    calc_paracousti_metrics(paracousti_directory, save_path, weights="All")

The result will be files with the same name as those in the paracousti_directory, but now in the save_directory. These files will have the updated netCDFs in them. 

.. note::
    All NetCDF in the paracouti_path will be processed. A timebar will be shown when running this function. Keep in mind that depending on the computer, this could take several hours.
