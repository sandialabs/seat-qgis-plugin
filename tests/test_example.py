import unittest
from qgis.core import QgsApplication
# Adjust the import path if necessary
# from seat.stressor_receptor_calc import classFactory


class ExampleTest(unittest.TestCase):
    def test_basic(self):
        self.assertTrue(True)


class TestPandasInstallation(unittest.TestCase):
    def test_pandas_installed(self):
        try:
            import pandas
            pandas_available = True
        except ImportError:
            pandas_available = False
        self.assertTrue(pandas_available, "Pandas is not installed.")


def run_all():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(ExampleTest))
    suite.addTest(unittest.makeSuite(TestPandasInstallation))
    runner = unittest.TextTestRunner()
    runner.run(suite)


if __name__ == '__main__':
    run_all()
