# -*- coding: utf-8 -*-
""" stressor_receptor_calc.py
 
 Copyright 2023, Integral Consulting Inc. All rights reserved.

 PURPOSE: A QGIS plugin that calculates a probability weighted response
    layer from stressor and/or receptor layers

 PROJECT INFORMATION:
 Name: SEAT - Spatial and Environmental Assessment Toolkit
 Number: C1308

 AUTHORS
 Eben Pendelton
  Timothy Nelson (tnelson@integral-corp.com)
  Caleb Grant (cgrant@inegral-corp.com)
  Sam McWilliams (smcwilliams@integral-corp.com)
 
 NOTES (Data descriptions and any script specific notes)
	1. plugin template from Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
	2. refer to documentation regarding installation and input formatting.
    3. requires installation of NETCDF4 (https://unidata.github.io/netcdf4-python/) 
       and QGIS (https://qgis.org/en/site/)
    4. tested and created using QGIS v3.22
    5. added habitat for shear stress
"""
import configparser
import logging
import os.path
import xml.etree.ElementTree as ET
from datetime import date
import numpy as np
import pandas as pd

from qgis.core import (
    Qgis,
    QgsApplication,
    QgsCoordinateReferenceSystem,
    QgsMessageLog,
    QgsProject,
    QgsRasterBandStats,
    QgsRasterLayer,
    QgsVectorLayer,
)
from qgis.gui import QgsProjectionSelectionDialog
from qgis.PyQt.QtCore import QCoreApplication, QSettings, QTranslator
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QFileDialog

# Initialize Qt resources from file resources.py
from .resources import *

# SEAT Modules
from .shear_stress_module import run_shear_stress_stressor
from .velocity_module import run_velocity_stressor
from .acoustics_module import run_acoustics_stressor
from .power_module import calculate_power

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
        locale = QSettings().value("locale/userLocale")
        if locale:
            locale = locale[0:2]
        else:
            locale = 'en'
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

        icon_path = ":/plugins/seat/icon.png"
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

    def select_style_files(self):
        """Input the styles_file file dialog."""
        filename, _filter = QFileDialog.getOpenFileName(
            self.dlg,
            "Select Style Files CSV file",
            "",
            "*.csv",
        )
        self.dlg.output_stylefile.setText(filename)

    def select_device_folder(self, presence):
        folder_name = QFileDialog.getExistingDirectory(
            self.dlg,
            "Select Folder"
        )
        if presence == "not present":
            self.dlg.device_not_present.setText(folder_name)
        else:
            self.dlg.device_present.setText(folder_name)

    def read_style_files(self, file):
        data = pd.read_csv(file)
        data = data.set_index('Type')
        return data

    def select_probabilities_file(self):
        """Input the probabilities file dialog."""
        filename, _filter = QFileDialog.getOpenFileName(
            self.dlg,
            "Select Probabilities File",
            "",
            "*.csv",
        )
        self.dlg.probabilities_file.setText(filename)

    def select_power_files_folder(self):
        folder_name = QFileDialog.getExistingDirectory(
            self.dlg,
            "Select Folder"
        )
        self.dlg.power_files.setText(folder_name)

    def select_crs(self):
        """Input the crs using the QGIS widget box."""

        proj_selector = QgsProjectionSelectionDialog(None)
        crs = QgsCoordinateReferenceSystem()
        crs.createFromId(4326)
        proj_selector.setCrs(crs)
        proj_selector.exec()
        self.dlg.crs.setText(proj_selector.crs().authid().split(":")[1])

    def select_receptor_file(self):
        """Input the receptor file."""
        filename, _filter = QFileDialog.getOpenFileName(
            self.dlg,
            "Select Receptor",
            "",
            "*.tif; *.csv",
        )
        self.dlg.receptor_file.setText(filename)

    def select_secondary_constraint_folder(self):
        """Select secondary constriant file."""
        folder_name = QFileDialog.getExistingDirectory(
            self.dlg,
            "Select Folder"
        )
        self.dlg.sc_file.setText(folder_name)

    def select_output_folder(self):
        """Select output file picker."""
        folder_name = QFileDialog.getExistingDirectory(
            self.dlg,
            "Select Folder"
        )
        self.dlg.output_folder.setText(folder_name)

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

            self.dlg.device_present.setText(
                config.get("Input", "device present filepath"),
            )
            self.dlg.device_not_present.setText(
                config.get("Input", "device not present filepath"),
            )
            self.dlg.probabilities_file.setText(config.get(
                "Input", "probabilities filepath"))
            self.dlg.power_files.setText(
                config.get("Input", "power files filepath"))

            self.dlg.receptor_file.setText(
                config.get("Input", "receptor filepath"))
            self.dlg.sc_file.setText(
                config.get("Input", "secondary constraint filepath"),
            )
            self.dlg.stressor_comboBox.setCurrentText(
                config.get("Input", "stressor variable"),
            )
            self.dlg.crs.setText(config.get(
                "Input", "coordinate reference system"))

            self.dlg.output_folder.setText(
                config.get("Output", "output filepath"))

            self.dlg.output_stylefile.setText(
                config.get("Input", "output style files"))

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
            "device present filepath": self.dlg.device_present.text(),
            "device not present filepath": self.dlg.device_not_present.text(),
            "probabilities filepath": self.dlg.probabilities_file.text(),
            "power files filepath": self.dlg.power_files.text(),
            "receptor filepath": self.dlg.receptor_file.text(),
            "secondary constraint filepath": self.dlg.sc_file.text(),
            "stressor variable": self.dlg.stressor_comboBox.currentText(),
            "coordinate reference system": self.dlg.crs.text(),
            "output style files": self.dlg.output_stylefile.text(),
        }

        config["Output"] = {"output filepath": self.dlg.output_folder.text()}

        with open(filename, "w") as configfile:
            config.write(configfile)

    def style_layer(self, fpath, stylepath, checked=True, ranges=True):
        """Style and add the result layer to map."""
        basename = os.path.splitext(os.path.basename(fpath))[0]
        layer = QgsProject.instance().addMapLayer(QgsRasterLayer(fpath, basename))

        if stylepath != "":
            # apply layer style
            layer.loadNamedStyle(stylepath)
            layer.triggerRepaint()
            # reload to see layer classification
            layer.reload()

        # refresh legend entries
        self.iface.layerTreeView().refreshLayerSymbology(layer.id())

        # do we want the layer visible in the map?
        if not checked:
            root = QgsProject.instance().layerTreeRoot()
            root.findLayer(layer.id()).setItemVisibilityChecked(checked)

        # do we want to return the ranges?
        if ranges:
            range = [x[0] for x in layer.legendSymbologyItems()]
            return range

    def run(self):
        """Run method that performs all the real work."""

        # Create the dialog with elements (after translation) and keep reference
        # Only create GUI ONCE in callback, so that it will only load when the plugin is started
        if self.first_start is True:
            self.first_start = False
            self.dlg = StressorReceptorCalcDialog()

            sfields = [
                "Shear Stress",
                "Velocity",
                "Acoustics"
            ]
            self.dlg.stressor_comboBox.addItems(sfields)

            # this connects the input file chooser
            self.dlg.load_input.clicked.connect(
                lambda: self.select_and_load_in())

            # this connects the input file creator
            self.dlg.save_input.clicked.connect(lambda: self.save_in())

            # set the present and not present files. Either .nc files or .tif folders
            self.dlg.device_pushButton.clicked.connect(
                lambda: self.select_device_folder("present"),
            )
            self.dlg.no_device_pushButton.clicked.connect(
                lambda: self.select_device_folder("not present"),
            )

            # set the probabilities
            self.dlg.probabilities_pushButton.clicked.connect(
                self.select_probabilities_file)

            self.dlg.power_files_pushButton.clicked.connect(
                self.select_power_files_folder)

            # set the crs file
            self.dlg.crs_button.clicked.connect(self.select_crs)

            # set the receptor file
            self.dlg.receptor_button.clicked.connect(self.select_receptor_file)

            # set the secondary constraint
            self.dlg.secondary_constraint_pushButton.clicked.connect(
                self.select_secondary_constraint_folder)
            # set the output
            self.dlg.output_pushButton.clicked.connect(
                self.select_output_folder)

            self.dlg.select_stylefile_button.clicked.connect(
                self.select_style_files)

        self.dlg.device_present.clear()
        self.dlg.device_not_present.clear()
        self.dlg.probabilities_file.clear()
        self.dlg.power_files.clear()
        self.dlg.crs.clear()
        self.dlg.receptor_file.clear()
        self.dlg.sc_file.clear()
        self.dlg.output_folder.clear()
        self.dlg.output_stylefile.clear()

        # show the dialog
        self.dlg.show()
        # Run the dialog event loop
        result = self.dlg.exec_()
        # See if OK was pressed
        if result:
            # Run Calculations
            # this grabs the files for input and output
            dpresentfname = self.dlg.device_present.text()
            dnotpresentfname = self.dlg.device_not_present.text()
            probabilities_fname = self.dlg.probabilities_file.text()
            power_files_folder = self.dlg.power_files.text()

            rfilename = self.dlg.receptor_file.text()
            scfilename = self.dlg.sc_file.text()
            output_folder_name = self.dlg.output_folder.text()
            # create output directory if it doesn't exist
            os.makedirs(output_folder_name, exist_ok=True)

            svar = self.dlg.stressor_comboBox.currentText()

            crs = int(self.dlg.crs.text())

            # create logger
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)

            # create file handler and set level to info
            fname = os.path.join(output_folder_name, "_{}.log".format(
                date.today().strftime("%Y%m%d")))
            fh = logging.FileHandler(fname, mode="a", encoding="utf8")
            fh.setLevel(logging.INFO)

            # create formatter
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )

            # add formatter to ch
            fh.setFormatter(formatter)

            # add ch to logger
            logger.addHandler(fh)

            logger.info("Device present File: %s", dpresentfname)
            logger.info("Device not present File: %s", dnotpresentfname)
            logger.info("Probabilities File: %s", probabilities_fname)
            logger.info("Power Files: %s", power_files_folder)
            logger.info("Stressor: %s", svar)
            logger.info("CRS: %s", crs)
            logger.info("Secondary Constraint File: %s", scfilename)
            logger.info("Output Folder: %s", output_folder_name)

            # if the output file path is empty display a warning
            if output_folder_name == "":
                QgsMessageLog.logMessage(
                    "Output file path not given.",
                    level=Qgis.MessageLevel.Warning,
                )

            # Calculate Power Files
            if power_files_folder is not "":
                logger.info("Power File Folder: %s", power_files_folder)
                calculate_power(power_files_folder, probabilities_fname,
                                save_path=output_folder_name,
                                crs=crs)

            if svar == "Shear Stress":

                if not ((scfilename is None) or (scfilename == "")):
                    scfilename = [os.path.join(scfilename, i) for i in os.listdir(
                        scfilename) if i.endswith('tif')][0]

                sfilenames = run_shear_stress_stressor(
                    dev_present_file=dpresentfname,
                    dev_notpresent_file=dnotpresentfname,
                    probabilities_file=probabilities_fname,
                    crs=crs,
                    output_path=output_folder_name,
                    receptor_filename=rfilename,
                    secondary_constraint_filename=scfilename
                )

                stylefiles_DF = self.read_style_files(
                    self.dlg.output_stylefile.text())

                sstylefile = stylefiles_DF.loc['Stressor'].values.item().replace(
                    "\\", "/")
                rstylefile = stylefiles_DF.loc['Receptor'].values.item().replace(
                    "\\", "/")
                scstylefile = stylefiles_DF.loc['Secondary Constraint'].values.item().replace(
                    "\\", "/")
                swrstylefile = stylefiles_DF.loc['Stressor with receptor'].values.item(
                ).replace("\\", "/")
                rcstylefile = stylefiles_DF.loc['Reclassificed Stressor with receptor'].values.item(
                ).replace("\\", "/")
                rskstylefile = stylefiles_DF.loc['Risk'].values.item(
                ).replace("\\", "/")

                logger.info("Receptor Style File: {}".format(rstylefile))
                logger.info("Stressor Style File: {}".format(sstylefile))
                logger.info(
                    "Secondary Constraint Style File: {}".format(scstylefile))
                logger.info("Output Style File: {}".format(
                    swrstylefile))  # stressor with receptor
                logger.info(
                    'Stressor reclassification: {}'.format(rcstylefile))
                logger.info(
                    'Risk: {}'.format(rskstylefile))
                srfilename = sfilenames['calculated_stressor']  # stressor
                self.style_layer(srfilename, sstylefile, ranges=True)
                # self.calc_area_change(srfilename, crs)
                if not ((rfilename is None) or (rfilename == "")):  # if receptor present
                    # streessor with receptor
                    self.style_layer(
                        sfilenames['calculated_stressor_with_receptor'], swrstylefile, ranges=True)
                    self.style_layer(
                        sfilenames['calculated_stressor_reclassified'], rcstylefile, ranges=True)  # reclassified
                    self.style_layer(
                        sfilenames['risk'], rskstylefile, ranges=True)
                    if rfilename.endswith('.tif'):
                        self.style_layer(rfilename, rstylefile, checked=False)

                if not ((scfilename is None) or (scfilename == "")):
                    if scfilename.endswith('.tif'):
                        self.style_layer(
                            sfilenames['secondary_constraint'], scstylefile, checked=False)

            if svar == "Velocity":
                if not ((scfilename is None) or (scfilename == "")):
                    scfilename = [os.path.join(scfilename, i) for i in os.listdir(
                        scfilename) if i.endswith('tif')][0]

                sfilenames = run_velocity_stressor(
                    dev_present_file=dpresentfname,
                    dev_notpresent_file=dnotpresentfname,
                    probabilities_file=probabilities_fname,
                    crs=crs,
                    output_path=output_folder_name,
                    receptor_filename=rfilename,
                    secondary_constraint_filename=scfilename
                )

                # sfilenames = ['calculated_stressor.tif',
                #  'calculated_stressor_with_receptor.tif',
                # 'calculated_stressor_reclassified.tif']

                stylefiles_DF = self.read_style_files(
                    self.dlg.output_stylefile.text())

                sstylefile = stylefiles_DF.loc['Stressor'].values.item().replace(
                    "\\", "/")
                rstylefile = stylefiles_DF.loc['Receptor'].values.item().replace(
                    "\\", "/")
                scstylefile = stylefiles_DF.loc['Secondary Constraint'].values.item().replace(
                    "\\", "/")
                swrstylefile = stylefiles_DF.loc['Stressor with receptor'].values.item(
                ).replace("\\", "/")
                rcstylefile = stylefiles_DF.loc['Reclassificed Stressor with receptor'].values.item(
                ).replace("\\", "/")

                logger.info("Receptor Style File: {}".format(rstylefile))
                logger.info("Stressor Style File: {}".format(sstylefile))
                logger.info(
                    "Secondary Constraint Style File: {}".format(scstylefile))
                logger.info("Output Style File: {}".format(
                    swrstylefile))  # stressor with receptor
                logger.info(
                    'Stressor reclassification: {}'.format(rcstylefile))

                srfilename = sfilenames['calculated_stressor']  # stressor
                self.style_layer(srfilename, sstylefile, ranges=True)
                # self.calc_area_change(srfilename, crs)
                if not ((rfilename is None) or (rfilename == "")):  # if receptor present
                    # streessor with receptor
                    swrfilename = sfilenames['calculated_stressor_with_receptor']
                    # reclassified
                    classifiedfilename = sfilenames['calculated_stressor_reclassified']
                    self.style_layer(swrfilename, swrstylefile, ranges=True)
                    self.style_layer(classifiedfilename,
                                     rcstylefile, ranges=True)
                    if rfilename.endswith('.tif'):
                        self.style_layer(rfilename, rstylefile, checked=False)

                if not ((scfilename is None) or (scfilename == "")):
                    if scfilename.endswith('.tif'):
                        self.style_layer(
                            sfilenames['secondary_constraint'], scstylefile, checked=False)

            if svar == "Acoustics":
                sfilenames = run_acoustics_stressor(
                    dev_present_file=dpresentfname,
                    dev_notpresent_file=dnotpresentfname,
                    probabilities_file=probabilities_fname,
                    crs=crs,
                    output_path=output_folder_name,
                    receptor_filename=rfilename,
                    species_folder=scfilename
                )  # TODO: Make output a dictionary for consistency
                # numpy_arrays = [0] PARACOUSTI
                #               [1] stressor
                #               [2] threshold_exceeded
                #               [3] percent_scaled
                #               [4] density_scaled

                stylefiles_DF = self.read_style_files(
                    self.dlg.output_stylefile.text())

                stressor_stylefile = stylefiles_DF.loc['Stressor'].values.item().replace(
                    "\\", "/")
                threshold_stylefile = stylefiles_DF.loc['Threshold'].values.item().replace(
                    "\\", "/")
                percent_stylefile = stylefiles_DF.loc['Species Percent'].values.item().replace(
                    "\\", "/")
                density_stylefile = stylefiles_DF.loc['Species Density'].values.item().replace(
                    "\\", "/")

                logger.info("Stressor Style File: {}".format(
                    stressor_stylefile))
                logger.info("Threshold Style File: {}".format(
                    threshold_stylefile))
                logger.info("Species Percent Style File: {}".format(
                    percent_stylefile))
                logger.info("Species Density Style File: {}".format(
                    density_stylefile))

                # self.calc_area_change(srfilename, crs)
                if not ((scfilename is None) or (scfilename == "")):  # if specie files present
                    self.style_layer(
                        sfilenames[4], density_stylefile, ranges=True)
                    self.style_layer(
                        sfilenames[3], percent_stylefile, ranges=True)

                self.style_layer(
                    sfilenames[2], threshold_stylefile, ranges=True)
                self.style_layer(
                    sfilenames[0], stressor_stylefile, ranges=True)  # paracousti
                self.style_layer(
                    sfilenames[1], stressor_stylefile, ranges=True)  # stressor

            # close and remove the filehandler
            fh.close()
            logger.removeHandler(fh)
