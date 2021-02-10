import model
import unittest


class TestDischargePolicy(unittest.TestCase):
    def test_simple(self):
        D = [4, 2, 4, 2]
        charge = 2
        schedule = model.discharge_policy(charge, D)
        self.assertListEqual(schedule, [1, 0, 1, 0])


if __name__ == "__main__":
    unittest.main()
