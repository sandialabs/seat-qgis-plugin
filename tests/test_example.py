# test_example.py

import unittest


class TestExample(unittest.TestCase):

    def test_basic(self):
        """
        A basic test that always passes.
        """
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
