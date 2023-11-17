import unittest


class ExampleTest(unittest.TestCase):
    def test_basic(self):
        self.assertTrue(True)


def run_all():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(ExampleTest))
    runner = unittest.TextTestRunner()
    runner.run(suite)


if __name__ == '__main__':
    run_all()
