"""
Methods

assertEqual(a, b)
assertNotEqual(a, b)
assertTrue(x)
assertFalse(x)
assertIs(a, b)
assertIsNot(a, b)
assertIsNone(x)
assertIsNotNone(x)
assertIn(a, b)
assertNotIn(a, b)
assertIsInstance(a, b)
assertNotIsInstance(a, b)
"""
import unittest
import trikit


class UtilsTestCase(unittest.TestCase):
    def setUp(self):
        self.datasets = ['amw09', 'autoliab', 'glre', 'raa', 'singinjury', 'singproperty', 'ta83',]
        self.lrdb_lobs = ['comauto', 'medmal', 'othliab', 'ppauto', 'prodliab', 'wkcomp']
        self.raa_incr_sum = 160987

    def test_datasets(self):
        datasets_list = trikit.get_datasets()
        self.assertTrue(
            all([ii==jj for ii,jj in zip(trikit.get_datasets(), self.datasets)]),
            "Non-equality between computed vs. datasets reference."
            )

    def test_lobs(self):
        datasets_list = trikit.get_lrdb_lobs()
        self.assertTrue(
            all([ii==jj for ii,jj in zip(trikit.get_lrdb_lobs(), self.lrdb_lobs)]),
            "Non-equality between computed vs. lrdb_lobs reference."
            )

    def test_raa2df(self):
        self.assertTrue(
            trikit.load("raa", tri_type=None).value.sum()==self.raa_incr_sum,
            "Non-equality between computed vs. raa losses."
            )

    def test_raa2incr(self):
        self.assertTrue(
            isinstance(trikit.load("raa", tri_type="incr"), trikit.triangle.IncrTriangle),
            "RAA dataset not coerced to incremental triangle object."
            )

    def test_raa2cum(self):
        self.assertTrue(
            isinstance(trikit.load("raa", tri_type="cum"), trikit.triangle.CumTriangle),
            "RAA dataset not coerced to cumulative triangle object."
            )


if __name__ == "__main__":

    unittest.main()