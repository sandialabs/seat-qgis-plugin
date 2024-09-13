.. _01_paracousti_preprocessing:
Paracousti Pre-Processing 
-------------------------
Several metrics can be calculated from ParAcousti output that are useful in determining impacts to marine species. These include the weighted and unweighted sound pressure level (SPL) and sound exposure level (SEL). Due to the resource and time-intensive calculations, a stand-alone python routine is included within the SEAT package. This routine calculates the user selected metrics and weights form the existing Paracousti metrics. The calculated metrics are saved to a netcdf variable with the same dimensions as the original Paracousti netcdf.
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

To run the Paracousti pre-processing routine,

.. code-block:: python

    from utils import paracousti_fxns 
    paracousti_path  = r'./Path/to/Paracousti/netCDF' 
    save_path = r'./Path/to/save/updated/netCDF'
    calc_paracousti_metrics(paracousti_path, save_path, weights="All")


.. note::
    All NetCDF in the paracouti_path will be processed. A timebar will be shown when running this function. Keep in mind that depending on the computer, this could take several hours.
