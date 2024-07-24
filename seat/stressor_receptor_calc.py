# -*- coding: utf-8 -*-
"""
/***************************************************************************.

 stressor_receptor_calc.py
 Copyright 2023, Integral Consulting Inc. All rights reserved.

 PURPOSE: A QGIS plugin that calculates a probability weighted response layer from stressor and/or receptor layers

 PROJECT INFORMATION:
 Name: SEAT - Spatial and Environmental Assessment Toolkit (https://github.com/sandialabs/seat-qgis-plugin)
 Number: C1308

 AUTHORS
 Eben Pendelton
  Timothy Nelson (tnelson@integral-corp.com)
  Caleb Grant (cgrant@inegral-corp.com)
  Sam McWilliams (smcwilliams@integral-corp.com)
 
 NOTES (Data descriptions and any script specific notes)
	1. plugin template from Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
	2. refer to documentation regarding installation and input formatting.
    3. requires installation of NETCDF4 (https://unidata.github.io/netcdf4-python/) and QGIS (https://qgis.org/en/site/)
    4. tested and created using QGIS v3.22
    5. added habitat for shear stress
"""
import configparser
# import logging
import os.path
import xml.etree.ElementTree as ET
# from datetime import date
# import numpy as np
import pandas as pd

from qgis.core import (# type: ignore
    Qgis,
    # QgsApplication,
    QgsCoordinateReferenceSystem,
    QgsMessageLog,
    QgsProject,
    # QgsRasterBandStats,
    QgsRasterLayer,
    # QgsVectorLayer,
    # QgsLayerTreeGroup,
    # QgsMapLayerStyleManager
)# type: ignore
from qgis.gui import QgsProjectionSelectionDialog  # ,QgsLayerTreeView # type: ignore
from qgis.PyQt.QtCore import QCoreApplication, QSettings, QTranslator  # type: ignore
from qgis.PyQt.QtGui import QIcon  # type: ignore
from qgis.PyQt.QtWidgets import QAction, QFileDialog  # type: ignore


# Initialize Qt resources from file resources.py
from .resources import *

# Import Modules
from .modules.shear_stress_module import run_shear_stress_stressor
from .modules.velocity_module import run_velocity_stressor
from .modules.acoustics_module import run_acoustics_stressor, find_acoustic_metrics
from .modules.power_module import calculate_power

# Import the code for the dialog
from .stressor_receptor_calc_dialog import StressorReceptorCalcDialog


# Most of the below is boilerplate code  until plugin specific functions start----
def df_from_qml(fpath):
    tree = ET.parse(fpath)
    root = tree.getroot()

    v = [i.get("label") for i in root[3][1][2][0].findall("item")]
    v2 = [s.split(" - ") for s in v]
    df = pd.DataFrame(v2, columns=["min", "max"])
    if df.empty:
        # grab the values in a unique palette
        v1 = [i.get("value") for i in root[3][1][2].findall("paletteEntry")]
        v2 = [i.get("label") for i in root[3][1][2].findall("paletteEntry")]
        df = pd.DataFrame({"value": v1, "label": v2})
    return df


class StressorReceptorCalc:
    """QGIS Plugin Implementation."""

    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """

        # Save reference to the QGIS interface
        self.iface = iface
        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)
        # initialize locale
        locale = QSettings().value("locale/userLocale")[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            "i18n",
            "StressorReceptorCalc_{}.qm".format(locale),
        )

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)
            QCoreApplication.installTranslator(self.translator)

        # Declare instance attributes
        self.actions = []
        self.menu = self.tr("&Stressor Receptor Calculator")

        # Check if plugin was started the first time in current QGIS session
        # Must be set in initGui() to survive plugin reloads
        self.first_start = None

    # noinspection PyMethodMayBeStatic
    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate("StressorReceptorCalc", message)

    def add_action(
        self,
        icon_path,
        setText,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None,
    ):
        """Add a toolbar icon to the toolbar.

        :param icon_path: Path to the icon for this action. Can be a resource
            path (e.g. ':/plugins/foo/bar.png') or a normal file system path.
        :type icon_path: str

        :param text: Text that should be shown in menu items for this action.
        :type text: str

        :param callback: Function to be called when the action is triggered.
        :type callback: function

        :param enabled_flag: A flag indicating if the action should be enabled
            by default. Defaults to True.
        :type enabled_flag: bool

        :param add_to_menu: Flag indicating whether the action should also
            be added to the menu. Defaults to True.
        :type add_to_menu: bool

        :param add_to_toolbar: Flag indicating whether the action should also
            be added to the toolbar. Defaults to True.
        :type add_to_toolbar: bool

        :param status_tip: Optional text to show in a popup when mouse pointer
            hovers over the action.
        :type status_tip: str

        :param parent: Parent widget for the new action. Defaults None.
        :type parent: QWidget

        :param whats_this: Optional text to show in the status bar when the
            mouse pointer hovers over the action.

        :returns: The action that was created. Note that the action is also
            added to self.actions list.
        :rtype: QAction
        """

        icon = QIcon(icon_path)
        action = QAction(icon, setText, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            # Adds plugin icon to Plugins toolbar
            self.iface.addToolBarIcon(action)

        if add_to_menu:
            self.iface.addPluginToMenu(
                self.menu,
                action,
            )

        self.actions.append(action)

        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""

        icon_path = ":/plugins/seat_qgis_plugin/icon.png"
        self.add_action(
            icon_path,
            setText=self.tr(
                "Calculate a response layer from stressor and receptor layers",
            ),
            callback=self.run,
            parent=self.iface.mainWindow(),
        )

        # will be set False in run()
        self.first_start = True

    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginMenu(
                self.tr("&Stressor Receptor Calculator"),
                action,
            )
            self.iface.removeToolBarIcon(action)

    # End mostly boilerplate code ------

    def select_folder(self):
        folder_name = QFileDialog.getExistingDirectory(
            self.dlg,
            "Select Folder"
        )
        return folder_name

    def read_style_files(self, file):
        data = pd.read_csv(file)
        data = data.set_index('Type')
        return data

    def select_file(self, filter=""):
        """Input the receptor file."""
        filename, _filter = QFileDialog.getOpenFileName(
            self.dlg,
            "Select File",
            "",
            filter,
        )
        return filename
    
    def copy_shear_input_to_velocity(self):
        self.dlg.velocity_device_present.setText(self.dlg.shear_device_present.text())
        self.dlg.velocity_device_not_present.setText(self.dlg.shear_device_not_present.text())            
        self.dlg.velocity_probabilities_file.setText(self.dlg.shear_probabilities_file.text())               
        self.dlg.velocity_risk_file.setText(self.dlg.shear_risk_file.text())
                
    def select_crs(self):
        """Input the crs using the QGIS widget box."""

        projSelector = QgsProjectionSelectionDialog(None)
        # set up a default one
        # crs = QgsCoordinateReferenceSystem()
        # crs.createFromId(4326)
        crs = QgsCoordinateReferenceSystem.fromWkt("EPSG:4326")
        projSelector.setCrs(crs)
        projSelector.exec()
        # projSelector.exec_()
        self.dlg.crs.setText(projSelector.crs().authid().split(":")[1])

    def test_exists(self, line_edit, fin, fin_type):
        if not ((fin is None) or (fin == "")):
            if os.path.exists(fin):
                line_edit.setText(fin)
                line_edit.setStyleSheet("color: black;")
            else:
                line_edit.setText(f'{fin_type} not Found')
                line_edit.setStyleSheet("color: red;")      
        else:
            line_edit.setStyleSheet("color: black;") 

    def select_and_load_in(self):
        """Select and load an input file."""
        filename, _filter = QFileDialog.getOpenFileName(
            self.dlg,
            "Select Input file",
            "",
            "*.ini",
        )
        # if there's a file selected try and parse it
        if filename != "":
            # try to parse the ini file
            config = configparser.ConfigParser()
            config.read(filename)

            fin = config.get("Input", "shear stress device present filepath")
            self.test_exists(self.dlg.shear_device_present, fin, 'Directory')    
            fin = config.get("Input", "shear stress device not present filepath")
            self.test_exists(self.dlg.shear_device_not_present, fin, 'Directory')      
            fin = config.get("Input", "shear stress probabilities file")
            self.test_exists(self.dlg.shear_probabilities_file, fin, 'File')       
            fin = config.get("Input", "shear stress grain size file")
            self.test_exists(self.dlg.shear_grain_size_file, fin, 'File')        
            fin = config.get("Input", "shear stress risk layer file")
            self.test_exists(self.dlg.shear_risk_file, fin, 'File')
            self.dlg.shear_averaging_combobox.setCurrentText(config.get("Input", "shear stress averaging"))
                                                                
            fin = config.get("Input", "velocity device present filepath")
            self.test_exists(self.dlg.velocity_device_present, fin, 'Directory')
            fin = config.get("Input", "velocity device not present filepath")
            self.test_exists(self.dlg.velocity_device_not_present, fin, 'Directory')
            fin=config.get("Input", "velocity probabilities file")
            self.test_exists(self.dlg.velocity_probabilities_file, fin, 'File')
            fin=config.get("Input", "velocity threshold file")
            self.test_exists(self.dlg.velocity_threshold_file, fin, 'File')
            fin=config.get("Input", "velocity risk layer file")
            self.test_exists(self.dlg.velocity_risk_file, fin, 'File')
            self.dlg.velocity_averaging_combobox.setCurrentText(config.get("Input", "velocity Averaging"))
                                                                
            fin=config.get("Input", "paracousti device present filepath")
            self.test_exists(self.dlg.paracousti_device_present, fin, 'Directory')
            fin = config.get("Input", "paracousti device not present filepath")
            self.test_exists(self.dlg.paracousti_device_not_present, fin, 'Directory')
            fin = config.get("Input", "paracousti probabilities file")
            self.test_exists(self.dlg.paracousti_probabilities_file, fin, 'File')
            fin = config.get("Input", "paracousti threshold file")
            self.test_exists(self.dlg.paracousti_threshold_file, fin, 'File')
            fin = config.get("Input", "paracousti risk layer file")
            self.test_exists(self.dlg.paracousti_risk_file, fin, 'File')
            fin = config.get("Input", "paracousti species filepath")
            self.test_exists(self.dlg.paracousti_species_directory, fin, 'Directory')
            self.dlg.paracousti_averaging_combobox.setCurrentText(config.get("Input", "paracousti averaging"))
            self.dlg.paracousti_metric_selection_combobox.setCurrentText(config.get("Input", "paracousti metric"))
            self.dlg.paracousti_weighting_combobox.setCurrentText(config.get("Input", "paracousti weighting"))
                                                
            fin = config.get("Input", "power files filepath")
            self.test_exists(self.dlg.power_files, fin, 'Directory')                
            fin = config.get("Input", "power probabilities file")
            self.test_exists(self.dlg.power_probabilities_file, fin, 'File')             

            self.dlg.crs.setText(config.get("Input", "coordinate reference system"))

            self.dlg.output_folder.setText(config.get("Output", "output filepath"))

            fin = config.get("Input", "output style files")
            self.test_exists(self.dlg.output_stylefile, fin, 'File')

        if 'config' in locals(): #prevents error if window to closed without running
            config.clear()

    def is_float(self, value):
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def save_in(self):
        """Select and save an input file."""
        filename, _filter = QFileDialog.getSaveFileName(
            self.dlg,
            "Save input file",
            "",
            "*.ini",
        )

        # try to parse the ini file
        config = configparser.ConfigParser()

        config["Input"] = {
            "shear stress device present filepath": self.dlg.shear_device_present.text(),
            "shear stress device not present filepath": self.dlg.shear_device_not_present.text(),
            "shear stress averaging": self.dlg.shear_averaging_combobox.currentText(),            
            "shear stress probabilities file": self.dlg.shear_probabilities_file.text(),            
            "shear stress grain size file": self.dlg.shear_grain_size_file.text(),
            "shear stress risk layer file": self.dlg.shear_risk_file.text(),
            "velocity device present filepath": self.dlg.velocity_device_present.text(),
            "velocity device not present filepath": self.dlg.velocity_device_not_present.text(),
            "velocity averaging": self.dlg.velocity_averaging_combobox.currentText(),
            "velocity probabilities file": self.dlg.velocity_probabilities_file.text(),
            "velocity threshold file": self.dlg.velocity_threshold_file.text(),
            "velocity risk layer file": self.dlg.velocity_risk_file.text(),                          
            "paracousti device present filepath": self.dlg.paracousti_device_present.text(),
            "paracousti device not present filepath": self.dlg.paracousti_device_not_present.text(),            
            "paracousti averaging": self.dlg.paracousti_averaging_combobox.currentText(),
            "paracousti probabilities file": self.dlg.paracousti_probabilities_file.text(),
            "paracousti threshold file": self.dlg.paracousti_threshold_file.text(),
            "paracousti risk layer file": self.dlg.paracousti_risk_file.text(),
            "paracousti species filepath" : self.dlg.paracousti_species_directory.text(),
            "paracousti metric" : self.dlg.paracousti_metric_selection_combobox.currentText(),
            "paracousti weighting" : self.dlg.paracousti_weighting_combobox.currentText(),
            "power files filepath": self.dlg.power_files.text(),
            "power probabilities file": self.dlg.power_probabilities_file.text(),
            "coordinate reference system": self.dlg.crs.text(),
            "output style files": self.dlg.output_stylefile.text(),
        }

        config["Output"] = {"output filepath": self.dlg.output_folder.text()}

        with open(filename, "w") as configfile:
            config.write(configfile)
            
    def add_layer(self, fpath, root=None, group=None): 
        basename = os.path.splitext(os.path.basename(fpath))[0]
        if group is not None:
            vlayer = QgsRasterLayer(fpath, basename)
            QgsProject.instance().addMapLayer(vlayer)
            layer = root.findLayer(vlayer.id())
            clone = layer.clone()
            group.insertChildNode(0, clone)
            root.removeChildNode(layer)
        else:
            layer = QgsProject.instance().addMapLayer(QgsRasterLayer(fpath, basename))        

    def style_layer(self, fpath, stylepath=None, root=None, group=None):#, ranges=True):
        """Style and add the result layer to map."""
        basename = os.path.splitext(os.path.basename(fpath))[0]
        if group is not None:
            vlayer = QgsRasterLayer(fpath, basename)
            QgsProject.instance().addMapLayer(vlayer)
            root = QgsProject.instance().layerTreeRoot()
            if stylepath is not None:
                vlayer.loadNamedStyle(stylepath)
                vlayer.triggerRepaint()
                vlayer.reload() 
            layer = root.findLayer(vlayer.id())
            clone = layer.clone()
            group.insertChildNode(0, clone)
            root.removeChildNode(layer)
        else:
            layer = QgsProject.instance().addMapLayer(QgsRasterLayer(fpath, basename))     
            layer.loadNamedStyle(stylepath)
            layer.triggerRepaint()
            layer.reload()            
        # refresh legend entries
            self.iface.layerTreeView().refreshLayerSymbology(layer.id())
            
    def update_weights(self):
        if not ((self.dlg.paracousti_device_present.text() is None) or (self.dlg.paracousti_device_present.text() == "")):
            if  os.path.exists(self.dlg.paracousti_device_present.text()):
                #read variables in a single .nc file to determine metrics present and weighting
                filelist = [i for i in os.listdir(self.dlg.paracousti_device_present.text()) if i.endswith(r".nc")]  
                weights, unweighted_vars, weigthed_vars = find_acoustic_metrics(os.path.join(self.dlg.paracousti_device_present.text(), filelist[0]))
                self.dlg.paracousti_metric_selection_combobox.clear()
                self.dlg.paracousti_weighting_combobox.clear()
                self.dlg.paracousti_weighting_combobox.addItems(weights)  

    def select_folder_module(self, module=None, option=None):
        directory = self.select_folder()
        if module=='shear':
            if option=='device_present':
                self.dlg.shear_device_present.setText(directory)
                self.dlg.shear_device_present.setStyleSheet("color: black;")
            if option=="device_not_present":
                self.dlg.shear_device_not_present.setText(directory)
                self.dlg.shear_device_not_present.setStyleSheet("color: black;")
        if module=='velocity':
            if option=='device_present':
                self.dlg.velocity_device_present.setText(directory)
                self.dlg.velocity_device_present.setStyleSheet("color: black;")
            if option=="device_not_present":
                self.dlg.velocity_device_not_present.setText(directory)
                self.dlg.velocity_device_not_present.setStyleSheet("color: black;")
        if module=='paracousti':
            if option=='device_present':
                self.dlg.paracousti_device_present.setText(directory)
                self.dlg.paracousti_device_present.setStyleSheet("color: black;")
                self.update_weights() 
            if option=="device_not_present":
                self.dlg.paracousti_device_not_present.setText(directory)  
                self.dlg.paracousti_device_not_present.setStyleSheet("color: black;")         
            if option=='species_directory':
                self.dlg.paracousti_species_directory.setText(directory)
                self.dlg.paracousti_species_directory.setStyleSheet("color: black;")
        if module=='power':
            self.dlg.power_files.setText(directory)
            self.dlg.power_files.setStyleSheet("color: black;")
        if module=='output':
            self.dlg.output_folder.setText(directory)      
            self.dlg.output_folder.setStyleSheet("color: black;")

    def select_files_module(self, module=None, option=None):
        if module=='shear':
            if option=='probabilities_file':
                file = self.select_file(filter="*.csv")
                self.dlg.shear_probabilities_file.setText(file)      
                self.dlg.shear_probabilities_file.setStyleSheet("color: black;")
            if option=="grain_size_file":
                file = self.select_file(filter="*.tif; *.csv")
                self.dlg.shear_grain_size_file.setText(file)      
                self.dlg.shear_grain_size_file.setStyleSheet("color: black;")
            if option=="risk_file":
                file = self.select_file(filter="*.tif")
                self.dlg.shear_risk_file.setText(file)       
                self.dlg.shear_risk_file.setStyleSheet("color: black;")                      
        if module=='velocity':
            if option=='probabilities_file':
                file = self.select_file(filter="*.csv")
                self.dlg.velocity_probabilities_file.setText(file)      
                self.dlg.velocity_probabilities_file.setStyleSheet("color: black;")
            if option=="thresholds":
                file = self.select_file(filter="*.tif; *.csv")
                self.dlg.velocity_threshold_file.setText(file)      
                self.dlg.velocity_threshold_file.setStyleSheet("color: black;")
            if option=="risk_file":
                file = self.select_file(filter="*.tif")
                self.dlg.velocity_risk_file.setText(file)        
                self.dlg.velocity_risk_file.setStyleSheet("color: black;")  
        if module=='paracousti':
            if option=='probabilities_file':
                file = self.select_file(filter="*.csv")
                self.dlg.paracousti_probabilities_file.setText(file)      
                self.dlg.paracousti_probabilities_file.setStyleSheet("color: black;")
            # if option=="thresholds":
            #     file = self.select_file(filter="*.csv")
            #     self.dlg.paracousti_threshold_file.setText(file)      
            #     self.dlg.paracousti_threshold_file.setStyleSheet("color: black;")
            if option=="risk_file":
                file = self.select_file(filter="*.tif")
                self.dlg.paracousti_risk_file.setText(file)         
                self.dlg.paracousti_risk_file.setStyleSheet("color: black;")
        if module=='power':
            file = self.select_file(filter="*.csv")
            self.dlg.power_probabilities_file.setText(file)      
            self.dlg.power_probabilities_file.setStyleSheet("color: black;")
        if module=='style_files':
            file = self.select_file(filter="*.csv")
            self.dlg.output_stylefile.setText(file)      
            self.dlg.output_stylefile.setStyleSheet("color: black;")

    def updateparacoustimetrics(self, index=None):
        self.dlg.paracousti_metric_selection_combobox.clear()
        if not(self.dlg.paracousti_device_present.text()==''):
            _, unweighted_vars, weigthed_vars = find_acoustic_metrics(os.path.join(self.dlg.paracousti_device_present.text(),
                [i for i in os.listdir(self.dlg.paracousti_device_present.text()) if i.endswith(r".nc")][0]))
            if self.dlg.paracousti_weighting_combobox.currentText() == "None": # disply unweighted variables
                self.dlg.paracousti_metric_selection_combobox.addItems(unweighted_vars)
            else: # display weighted variables
                self.dlg.paracousti_metric_selection_combobox.addItems(weigthed_vars)

    def updateparacoustiunits(self, value=None):
        if 'spl'.casefold() in self.dlg.paracousti_metric_selection_combobox.currentText().casefold():
            self.dlg.paracousti_threshold_units.setText("dB re 1 μPa")
        elif 'sel'.casefold() in  self.dlg.paracousti_metric_selection_combobox.currentText().casefold():
            self.dlg.paracousti_threshold_units.setText("dB re 1 μPa²∙s")
            

    def checkparacoustithreshold(self):
        if self.is_float(self.dlg.paracousti_threshold_value.text()):
            self.dlg.paracousti_threshold_value.setStyleSheet("color: black;")
        else:
            print(f"\'{self.dlg.paracousti_threshold_value.text()}\' is not a valid threshold (must be a number).")
            self.dlg.setText(f"{self.dlg.paracousti_threshold_value.text()} is not a valid threshold (must be a number).")
            self.dlg.paracousti_threshold_value.setStyleSheet("color: red;")

    def checkparacoustiresolution(self):
        if self.is_float(self.dlg.paracousti_species_grid_resolution.text()):
            self.dlg.paracousti_species_grid_resolution.setStyleSheet("color: black;")
        else:
            print(f"\'{self.dlg.paracousti_species_grid_resolution.text()}\' is not a valid resolution (must be a number).")
            self.dlg.setText(f"{self.dlg.paracousti_species_grid_resolution.text()} is not a valid resolution (must be a number).")
            self.dlg.paracousti_species_grid_resolution.setStyleSheet("color: red;")
                
    def run(self):
        """Run method that performs all the real work."""

        # Create the dialog with elements (after translation) and keep reference
        # Only create GUI ONCE in callback, so that it will only load when the plugin is started
        if self.first_start is True:
            self.first_start = False
            self.dlg = StressorReceptorCalcDialog()

            shear_average_fields = [
                "Maximum",
                "Mean",
                "Final Timestep"
            ]
            self.dlg.shear_averaging_combobox.addItems(shear_average_fields)

            velocity_average_fields = [
                "Maximum",
                "Mean",
                "Final Timestep"
            ]
            self.dlg.velocity_averaging_combobox.addItems(velocity_average_fields)

            paracousti_average_fields = [
                "Depth Maximum",
                "Depth Average",
                "Bottom Bin",
                "Top Bin"
            ]
            self.dlg.paracousti_averaging_combobox.addItems(paracousti_average_fields)

            paracousti_weighting_fields = [] # empty until directory selected
            self.dlg.paracousti_weighting_combobox.addItems(paracousti_weighting_fields)       

            paracousti_metric_fields = [] # empty until directory selected
            self.dlg.paracousti_metric_selection_combobox.addItems(paracousti_metric_fields)    

            # this connects the input file chooser
            self.dlg.load_input.clicked.connect(
                lambda: self.select_and_load_in())

            # this connects the input file creator
            self.dlg.save_input.clicked.connect(lambda: self.save_in())

            # set the present and not present files. Either .nc files or .tif folders
            
            #directories
            self.dlg.shear_device_pushButton.clicked.connect(lambda: self.select_folder_module(module="shear", option="device_present"))
            self.dlg.shear_no_device_pushButton.clicked.connect(lambda: self.select_folder_module(module="shear", option="device_not_present"))
            self.dlg.velocity_device_pushButton.clicked.connect(lambda: self.select_folder_module(module="velocity", option="device_present"))
            self.dlg.velocity_no_device_pushButton.clicked.connect(lambda: self.select_folder_module(module="velocity", option="device_not_present"))           
            self.dlg.paracousti_no_device_pushButton.clicked.connect(lambda: self.select_folder_module(module="paracousti", option="device_not_present"))
            self.dlg.paracousti_device_pushButton.clicked.connect(lambda: self.select_folder_module(module="paracousti", option="device_present"))
            self.dlg.paracousti_species_directory_button.clicked.connect(lambda: self.select_folder_module(module="paracousti", option="species_directory"))
            self.dlg.power_files_pushButton.clicked.connect(lambda: self.select_folder_module(module="power"))
            self.dlg.output_pushButton.clicked.connect(lambda: self.select_folder_module(module="output"))
            
            #files
            self.dlg.shear_probabilities_pushButton.clicked.connect(lambda: self.select_files_module(module='shear', option='probabilities_file'))
            self.dlg.shear_grain_size_button.clicked.connect(lambda: self.select_files_module(module='shear', option='grain_size_file'))
            self.dlg.shear_risk_pushButton.clicked.connect(lambda: self.select_files_module(module='shear', option='risk_file'))
            self.dlg.velocity_probabilities_pushButton.clicked.connect(lambda: self.select_files_module(module='velocity', option='probabilities_file'))
            self.dlg.velocity_threshold_button.clicked.connect(lambda: self.select_files_module(module='velocity', option='thresholds'))
            self.dlg.velocity_risk_pushButton.clicked.connect(lambda: self.select_files_module(module='velocity', option='risk_file'))                
            self.dlg.paracousti_probabilities_pushButton.clicked.connect(lambda: self.select_files_module(module='paracousti', option='probabilities_file'))
            # self.dlg.paracousti_threshold_button.clicked.connect(lambda: self.select_files_module(module='paracousti', option='thresholds'))
            self.dlg.paracousti_risk_pushButton.clicked.connect(lambda: self.select_files_module(module='paracousti', option='risk_file'))                
            self.dlg.power_probabilities_pushButton.clicked.connect(lambda: self.select_files_module(module='power'))     
            self.dlg.select_stylefile_button.clicked.connect(lambda: self.select_files_module(module='style_files'))   
            
            self.dlg.copy_shear_to_velocity_button.clicked.connect(self.copy_shear_input_to_velocity)  
            self.dlg.crs_button.clicked.connect(self.select_crs)

            self.dlg.paracousti_device_present.textChanged.connect(lambda: self.update_weights())
            self.dlg.paracousti_weighting_combobox.currentIndexChanged.connect(lambda: self.updateparacoustimetrics())
            self.dlg.paracousti_metric_selection_combobox.currentIndexChanged.connect(lambda: self.updateparacoustiunits())
            self.dlg.paracousti_threshold_value.textChanged.connect(lambda: self.checkparacoustithreshold())
            self.dlg.paracousti_species_grid_resolution.textChanged.connect(lambda: self.checkparacoustiresolution())

        self.dlg.shear_device_present.clear()
        self.dlg.velocity_device_present.clear()
        self.dlg.paracousti_device_present.clear()
        self.dlg.power_files.clear()
        
        self.dlg.shear_device_not_present.clear()
        self.dlg.velocity_device_not_present.clear()        
        self.dlg.paracousti_device_not_present.clear()

        self.dlg.shear_probabilities_file.clear()
        self.dlg.velocity_probabilities_file.clear()
        self.dlg.paracousti_probabilities_file.clear()   
        self.dlg.power_probabilities_file.clear()                        

        self.dlg.shear_grain_size_file.clear()
        self.dlg.velocity_threshold_file.clear()
        self.dlg.paracousti_threshold_value.clear()
        self.dlg.paracousti_species_grid_resolution.clear()
        
        self.dlg.shear_risk_file.clear()
        self.dlg.velocity_risk_file.clear()
        self.dlg.paracousti_risk_file.clear()

        self.dlg.paracousti_species_directory.clear()
        
        self.dlg.crs.clear() 
        self.dlg.output_folder.clear()
        self.dlg.output_stylefile.clear()

        # show the dialog
        self.dlg.show()
        # Run the dialog event loop
        result = self.dlg.exec_()
        # See if OK was pressed
        if result:
            # Run Calculations
            # Run Calculations
            # this grabs the files for input and output
            #TODO Remove these and just query the dlg directly when needed
            shear_stress_device_present_directory = self.dlg.shear_device_present.text()
            if not ((shear_stress_device_present_directory is None) or (shear_stress_device_present_directory == "")):
                if not os.path.exists(shear_stress_device_present_directory):
                    raise FileNotFoundError(f"The directory {shear_stress_device_present_directory} does not exist.")
            velocity_device_present_directory = self.dlg.velocity_device_present.text()
            if not ((velocity_device_present_directory is None) or (velocity_device_present_directory == "")):
                if not os.path.exists(velocity_device_present_directory):
                    raise FileNotFoundError(f"The directory {velocity_device_present_directory} does not exist.")            
            paracousti_device_present_directory = self.dlg.paracousti_device_present.text()
            if not ((paracousti_device_present_directory is None) or (paracousti_device_present_directory == "")):
                if not os.path.exists(paracousti_device_present_directory):
                    raise FileNotFoundError(f"The directory {paracousti_device_present_directory} does not exist.")      
            power_files_directory = self.dlg.power_files.text()
            if not ((power_files_directory is None) or (power_files_directory == "")):
                if not os.path.exists(power_files_directory):
                    raise FileNotFoundError(f"The directory {power_files_directory} does not exist.")    
                            
            shear_stress_device_not_present_directory = self.dlg.shear_device_not_present.text()
            if not ((shear_stress_device_not_present_directory is None) or (shear_stress_device_not_present_directory == "")):
                if not os.path.exists(shear_stress_device_not_present_directory):
                    raise FileNotFoundError(f"The directory {shear_stress_device_not_present_directory} does not exist.")
            velocity_device_not_present_directory = self.dlg.velocity_device_not_present.text()
            if not ((velocity_device_not_present_directory is None) or (velocity_device_not_present_directory == "")):
                if not os.path.exists(velocity_device_not_present_directory):
                    raise FileNotFoundError(f"The directory {velocity_device_not_present_directory} does not exist.")
            paracousti_device_not_present_directory = self.dlg.paracousti_device_not_present.text()
            if not ((paracousti_device_not_present_directory is None) or (paracousti_device_not_present_directory == "")):
                if not os.path.exists(paracousti_device_not_present_directory):
                    raise FileNotFoundError(f"The directory {paracousti_device_not_present_directory} does not exist.")
            
            shear_stress_probabilities_fname = self.dlg.shear_probabilities_file.text()
            if not ((shear_stress_probabilities_fname is None) or (shear_stress_probabilities_fname == "")):
                if not os.path.exists(shear_stress_probabilities_fname):
                    raise FileNotFoundError(f"The file {shear_stress_probabilities_fname} does not exist.")
            velocity_probabilities_fname = self.dlg.velocity_probabilities_file.text()
            if not ((velocity_probabilities_fname is None) or (velocity_probabilities_fname == "")):
                if not os.path.exists(velocity_probabilities_fname):
                    raise FileNotFoundError(f"The file {velocity_probabilities_fname} does not exist.")
            paracousti_probabilities_fname = self.dlg.paracousti_probabilities_file.text()
            if not ((paracousti_probabilities_fname is None) or (paracousti_probabilities_fname == "")):
                if not os.path.exists(paracousti_probabilities_fname):
                    raise FileNotFoundError(f"The file {paracousti_probabilities_fname} does not exist.")
            power_probabilities_fname = self.dlg.power_probabilities_file.text()     
            if not ((power_probabilities_fname is None) or (power_probabilities_fname == "")):
                if not os.path.exists(power_probabilities_fname):
                    raise FileNotFoundError(f"The file {power_probabilities_fname} does not exist.")       

            shear_grain_size_file = self.dlg.shear_grain_size_file.text()     
            if not ((shear_grain_size_file is None) or (shear_grain_size_file == "")):
                if not os.path.exists(shear_grain_size_file):
                    raise FileNotFoundError(f"The file {shear_grain_size_file} does not exist.")   
            velocity_threshold_file = self.dlg.velocity_threshold_file.text()     
            if not ((velocity_threshold_file is None) or (velocity_threshold_file == "")):
                if not os.path.exists(velocity_threshold_file):
                    raise FileNotFoundError(f"The file {velocity_threshold_file} does not exist.") 
            shear_risk_layer_file = self.dlg.shear_risk_file.text()     
            if not ((shear_risk_layer_file is None) or (shear_risk_layer_file == "")):
                if not os.path.exists(shear_risk_layer_file):
                    raise FileNotFoundError(f"The file {shear_risk_layer_file} does not exist.")   
            velocity_risk_layer_file = self.dlg.velocity_risk_file.text()     
            if not ((velocity_risk_layer_file is None) or (velocity_risk_layer_file == "")):
                if not os.path.exists(velocity_risk_layer_file):
                    raise FileNotFoundError(f"The file {velocity_risk_layer_file} does not exist.")   
            paracousti_risk_layer_file = self.dlg.paracousti_risk_file.text()         
            if not ((paracousti_risk_layer_file is None) or (paracousti_risk_layer_file == "")):
                if not os.path.exists(paracousti_risk_layer_file):
                    raise FileNotFoundError(f"The file {paracousti_risk_layer_file} does not exist.")           

            paracousti_species_directory = self.dlg.paracousti_species_directory.text()         
            if not ((paracousti_species_directory is None) or (paracousti_species_directory == "")):
                if not os.path.exists(paracousti_species_directory):
                    raise FileNotFoundError(f"The directory {paracousti_species_directory} does not exist.")  

            paracousti_weighting = self.dlg.paracousti_weighting_combobox.currentText()
            paracousti_metric = self.dlg.paracousti_metric_selection_combobox.currentText()
            paracousti_threshold_value = self.dlg.paracousti_threshold_value.currentText()
            if self.is_float(paracousti_threshold_value):
                paracousti_threshold_value = float(paracousti_threshold_value)
                
            paracousti_species_grid_resolution = self.dlg.paracousti_species_grid_resolution.currentText()
            if self.is_float(paracousti_species_grid_resolution):
                paracousti_species_grid_resolution = float(paracousti_species_grid_resolution)
                            
            shear_stress_averaging = self.dlg.shear_averaging_combobox.currentText()              
            velocity_averaging = self.dlg.velocity_averaging_combobox.currentText()                   
            paracousti_averaging = self.dlg.paracousti_averaging_combobox.currentText()                   
            
            output_folder_name = self.dlg.output_folder.text()
            if not ((output_folder_name is None) or (output_folder_name == "")):
                os.makedirs(output_folder_name, exist_ok=True) # create output directory if it doesn't exist

            crs = int(self.dlg.crs.text())
            
            # need to add check to leave empty if not present then apply default values
            if not ((self.dlg.output_stylefile.text() is None) or (self.dlg.output_stylefile.text() == "")):
                if not os.path.exists(self.dlg.output_stylefile.text()):
                    raise FileNotFoundError(f"The file {self.dlg.output_stylefile.text()} does not exist.")
                stylefiles_DF = self.read_style_files(self.dlg.output_stylefile.text())
            else:
                stylefiles_DF = None

            initialize_group  = True

            # if the output file path is empty display a warning
            if output_folder_name == "":
                QgsMessageLog.logMessage("Output file path not given.", level=Qgis.MessageLevel.Warnin)

            # Run Power Module
            if power_files_directory != "":
                if ((power_probabilities_fname is None) or (power_probabilities_fname == "")):
                    power_probabilities_fname = shear_stress_probabilities_fname #default to shear stress probabilities if none given
                calculate_power(power_files_directory, 
                                power_probabilities_fname,
                                save_path=os.path.join(output_folder_name, 'Power Module'),
                                crs=crs)

            # Run Shear Stress Module 
            if not ((shear_stress_device_present_directory is None) or (shear_stress_device_present_directory == "")): # svar == "Shear Stress":
                sfilenames = run_shear_stress_stressor(
                    dev_present_file=shear_stress_device_present_directory,
                    dev_notpresent_file=shear_stress_device_not_present_directory,
                    probabilities_file=shear_stress_probabilities_fname,
                    crs=crs,
                    output_path=os.path.join(output_folder_name, 'Shear Stress Module'),
                    receptor_filename=shear_grain_size_file,
                    secondary_constraint_filename=shear_risk_layer_file,
                    value_selection=shear_stress_averaging)

                if initialize_group:
                    root = QgsProject.instance().layerTreeRoot()
                    group = root.addGroup("temporary")
                    self.add_layer(sfilenames[list(sfilenames.keys())[0]], root=root, group=group)
                    initialize_group = False

                group_name = "Shear Stress Stressor"
                root = QgsProject.instance().layerTreeRoot()
                group = root.findGroup(group_name)
                if group is None:
                    group = root.addGroup(group_name)
                for key in sfilenames.keys(): #add styles files and/or display
                    if stylefiles_DF is None:
                        self.add_layer(sfilenames[key], root=root, group=group)
                    else:
                        self.style_layer(sfilenames[key], stylefiles_DF.loc[key].item(), root=root, group=group)

            # Run Velocity Module
            if not ((velocity_device_present_directory is None) or (velocity_device_present_directory == "")): # svar == "Velocity":

                vfilenames = run_velocity_stressor(
                    dev_present_file=velocity_device_present_directory,
                    dev_notpresent_file=velocity_device_not_present_directory,
                    probabilities_file=velocity_probabilities_fname,
                    crs=crs,
                    output_path=os.path.join(output_folder_name, 'Velocity Module'),
                    receptor_filename=velocity_threshold_file,
                    secondary_constraint_filename=velocity_risk_layer_file,
                    value_selection=velocity_averaging)

                if initialize_group:
                    root = QgsProject.instance().layerTreeRoot()
                    group = root.addGroup("temporary")
                    self.add_layer(vfilenames[list(vfilenames.keys())[0]], root=root, group=group)
                    initialize_group = False                
                
                group_name = "Velocity Stressor"
                root = QgsProject.instance().layerTreeRoot()
                group = root.findGroup(group_name)
                if group is None:
                    group = root.addGroup(group_name)
                for key in vfilenames.keys(): #add styles files and/or display
                    if stylefiles_DF is None:
                        self.add_layer(vfilenames[key], root=root, group=group)
                    else:
                        self.style_layer(vfilenames[key] , stylefiles_DF.loc[key].item(), root=root, group=group)                 

            # Run Acoustics Module
            if not ((paracousti_device_present_directory is None) or (paracousti_device_present_directory == "")): # if svar == "Acoustics":

                pfilenames_probabilistic, pfilenames_nonprobabilistic = run_acoustics_stressor(
                    dev_present_file=paracousti_device_present_directory,
                    dev_notpresent_file=paracousti_device_not_present_directory,
                    probabilities_file=paracousti_probabilities_fname,
                    crs=crs,
                    output_path=os.path.join(output_folder_name, 'Acoustics Module'),
                    receptor_filename=paracousti_threshold_value,
                    paracousti_weighting=paracousti_weighting,
                    paracousti_metric=paracousti_metric,
                    species_folder=paracousti_species_directory,
                    paracousti_species_grid_resolution=paracousti_species_grid_resolution,
                    Averaging = paracousti_averaging,
                    secondary_constraint_filename=paracousti_risk_layer_file)
                
                if initialize_group:
                    root = QgsProject.instance().layerTreeRoot()
                    group = root.addGroup("temporary")
                    self.add_layer(pfilenames_probabilistic[list(pfilenames_probabilistic.keys())[0]], root=root, group=group)
                    initialize_group = False
                
                group_name = "Acoustic Stressor - Probabilistic"
                root = QgsProject.instance().layerTreeRoot()
                group = root.findGroup(group_name)
                if group is None:
                    group = root.addGroup(group_name)
                for key in pfilenames_probabilistic.keys(): #add styles files and/or display

                    if stylefiles_DF is None:
                        self.add_layer(pfilenames_probabilistic[key], root=root, group=group)
                    else:
                        self.style_layer(pfilenames_probabilistic[key] , stylefiles_DF.loc[key].item(), root=root, group=group)

                group_name = "Acoustic Stressor - Non-Probilistic"
                root = QgsProject.instance().layerTreeRoot()
                group = root.findGroup(group_name)
                if group is None:
                    group = root.addGroup(group_name)
                for key in pfilenames_nonprobabilistic.keys(): #add styles files and/or display
                    for var in pfilenames_nonprobabilistic[key]:
                        if stylefiles_DF is None:
                            self.add_layer(pfilenames_nonprobabilistic[key][var], root=root, group=group)
                        else:
                            self.style_layer(pfilenames_nonprobabilistic[key][var], stylefiles_DF.loc[key].item(), root=root, group=group)                        

            #remove temproary layer group
            root = QgsProject.instance().layerTreeRoot()
            group_layer = root.findGroup("temporary")
            if group_layer is not None:
                root.removeChildNode(group_layer)
            
            # close and remove the filehandler
        # fh.close()