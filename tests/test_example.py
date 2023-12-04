import sys
import os
import unittest
import numpy as np
from qgis.core import QgsApplication

# Import seat
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
from seat import classFactory

from seat.shear_stress_module import critical_shear_stress

# Mock Interface
class MockIface:
    pass


class TestQGISPlugin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize QGIS Application
        qgis_install_path = os.environ.get('QGIS_ROOT', 'C:\\Program Files\\QGIS 3.34.1')
        QgsApplication.setPrefixPath(qgis_install_path, True)
        cls.qgs = QgsApplication([], False)
        cls.qgs.initQgis()
        cls.iface = MockIface()
        cls.plugin = classFactory(cls.iface)

    @classmethod
    def tearDownClass(cls):
        # Stop the QGIS application
        cls.qgs.exitQgis()

    def test_tr_method(self):
        test_string = "hello"
        expected_translation = "hello"
        result = self.plugin.tr(test_string)
        self.assertEqual(result, expected_translation)

    def test_critical_shear_stress(self):
        # Test with default parameters
        D_meters = np.array([0.001, 0.002])
        expected_output = np.array([0.52054529, 1.32156154]) 

        result = critical_shear_stress(D_meters)
        np.testing.assert_almost_equal(result, expected_output, decimal=5)


def run_all():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestQGISPlugin))
    runner = unittest.TextTestRunner()
    runner.run(suite)

if __name__ == '__main__':
    run_all()
