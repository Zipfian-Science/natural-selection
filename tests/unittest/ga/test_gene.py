import unittest
import random
from natural_selection.ga import Gene

class TestGene(unittest.TestCase):

    def test_init(self):
        g = Gene("test", 3, 10, 1, random.randint)
        self.assertEquals(str(g), '(test:3)')

        g_2 = g.randomize()

        self.assertNotEqual(g, g_2)