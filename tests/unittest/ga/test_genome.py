import unittest
import random
from natural_selection.ga import Gene, Genome

class TestGenome(unittest.TestCase):

    def test_init(self):
        g = Gene("test", 3, 10, 1, random.randint)
        gen = Genome([g])

        self.assertGreater(len(gen.genes), 0)

    def test_append(self):
        g = Gene("test", 3, 10, 1, random.randint)
        gen = Genome()
        gen.append(g)

        with self.assertRaises(AssertionError) as ae:
            gen.append(1)

    def test_indexing(self):
        g = Gene("test", 3, 10, 1, random.randint)
        gen = Genome([g])

        self.assertEquals(gen[0], g)

        with self.assertRaises(AssertionError) as ae:
            gen[2]

        with self.assertRaises(AssertionError) as ae:
            gen[0] = 1

        gen[0] = Gene("test_other", 3, 10, 1, random.randint)

        self.assertNotEqual(gen[0], g)

    def test_len_and_str_and_iter(self):
        g_1 = Gene("test", 3, 10, 1, random.randint)
        g_2 = Gene("test_other", 4, 10, 1, random.randint)
        gen = Genome([g_1, g_2])

        self.assertEquals(len(gen), 2)
        self.assertEquals(str(gen), '(test:3)-(test_other:4)')

        for g in gen:
            self.assertIsInstance(g, Gene)