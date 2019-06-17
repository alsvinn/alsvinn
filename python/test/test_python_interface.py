import unittest
import alsvinn
class TestPythonInterface(unittest.TestCase):

    def test_run(self):
        alsvinn_object = alsvinn.run(dimension=[8, 1, 1],
                    samples=1)
        data = alsvinn_object.get_data('rho', 0)
        self.assertEqual(data[0], 1.0)
        self.assertEqual(data[-1], 0.125)

if __name__ == '__main__':
    unittest.main()
