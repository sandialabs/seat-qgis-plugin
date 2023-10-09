Tanana River
------------

The Tanana River located in central Alaska is being considered as a viable site for current energy capture. The town of Nenana, located on the inner bank of a bend in the river, could benefit from locally generated energy rather than diesel generators. A current energy capture device has been tested on site and ongoing data collection events, related to local fish populations and benthic characteristics, could allow for a robust characterization of the impact a long-term deployment could have on the system. A model was developed on an unstructured grid in DFlow-FM to simulate a range of flow return periods with device arrays present. The flow cases are applied to both baseline (no devices present) conditions and with devices present. 


Sedimentation Analysis
^^^^^^^^^^^^^^^^^^^^^^

- Example model input can be found in "./DEMO unstructured/shear_stress_with_receptor_demo.ini"
  
  The model data consists of individual .nc files for each flow return period. The flow period within .nc filename is used to determine the probability of occurrence.

  * This set of inputs evaluates the impact on sediment mobility given a single median grain size receptor in a CSV file.
  
For this case the probability weighted shear stress for model runs with devices is compared to the probability weighted shear stress without devices.

.. figure:: ../media/cec_tutorial_shear_inputs.webp
   :scale: 100 %
   :alt: Tanana sedimentation example input


The calculated stressor (change in shear stress difference), stressor with receptor (change in sediment mobility), and the reclassified stressor from the analysis are shown below.


.. figure:: ../media/cec_tutorial_shear_risk.webp
   :alt: Tanana sedimentation example risk


Larval Transport Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^

- Example model input can be found in "DEMO unstructured/velocity_with_receptor_demo.ini"

  * This set of inputs evaluates the impact on larval motility given a single critical velocity receptor in a CSV file.
  
  For this case the velocity with devices is compared to the velocity without devices and a difference (stressor) is calculated.


.. figure:: ../media/cec_tutorial_velocity_risk.webp
   :scale: 100 %
   :alt: Tanana velocity example risk