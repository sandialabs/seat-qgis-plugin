import unittest
from qgis.core import QgsApplication
# Adjust the import path if necessary
from seat.stressor_receptor_calc import classFactory


class TestSeatPluginInstallation(unittest.TestCase):

    def setUp(self):
        self.qgis_app = QgsApplication([], False)
        QgsApplication.initQgis()

    def tearDown(self):
        QgsApplication.exitQgis()

    def test_plugin_initialization(self):
        mock_iface = None  # Mock interface, adjust as needed
        plugin_instance = classFactory(mock_iface)
        self.assertIsNotNone(plugin_instance)


class ExampleTest(unittest.TestCase):
    def test_basic(self):
        self.assertTrue(True)


def run_all():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(ExampleTest))
    suite.addTest(unittest.makeSuite(TestSeatPluginInstallation))
    runner = unittest.TextTestRunner()
    runner.run(suite)


if __name__ == '__main__':
    run_all()
