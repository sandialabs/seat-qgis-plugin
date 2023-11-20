from seat import classFactory
import unittest
from qgis.core import QgsApplication


# Mock Interface
class MockIface:
    # Add any methods here that your plugin expects from iface
    pass


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


class TestQGISPlugin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize QGIS Application
        cls.qgs = QgsApplication([], False)
        QgsApplication.initQgis()
        cls.iface = MockIface()
        cls.plugin = classFactory(cls.iface)

    @classmethod
    def tearDownClass(cls):
        # Stop the QGIS application
        QgsApplication.exitQgis()

    def test_translation_function(self):
        translated = self.plugin.tr('hello')
        self.assertIsNotNone(translated, "Translation function returned None")


def run_all():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(ExampleTest))
    suite.addTest(unittest.makeSuite(TestPandasInstallation))
    suite.addTest(unittest.makeSuite(TestQGISPlugin))
    runner = unittest.TextTestRunner()
    runner.run(suite)


if __name__ == '__main__':
    run_all()
