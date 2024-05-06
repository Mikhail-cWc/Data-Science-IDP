import unittest
from score_function import score_fn, error_by_category

class TestScoreFunction(unittest.TestCase):

    def test_empty_lists(self):
        self.assertEqual(score_fn([], [], 1.0), 0.0)

    def test_precision_recall(self):
        # precision = 1
        # recall = 2/3
        gold = ['entity1', 'entity2', 'entity3']
        pred = ['entity1', 'entity3']
        self.assertEqual(score_fn(gold, pred, 1.0), 0.8)

    def test_zero_precision(self):
        gold = ['entity1', 'entity2', 'entity3']
        pred = []
        self.assertEqual(score_fn(gold, pred, 1.0), 0.0)

    def test_zero_recall(self):
        gold = []
        pred = ['entity1', 'entity2', 'entity3']
        self.assertEqual(score_fn(gold, pred, 1.0), 0.0)

    def test_beta_less_than_zero(self):
        gold = ['entity1', 'entity2', 'entity3']
        pred = ['entity1', 'entity3']
        with self.assertRaises(ValueError):
            score_fn(gold, pred, -1.0)

    def test_beta_equal_to_zero(self):
        gold = ['entity1', 'entity2', 'entity3']
        pred = ['entity1', 'entity3']
        with self.assertRaises(ValueError):
            score_fn(gold, pred, 0.0)

    def test_beta_greater_than_one(self):
        # precision = 1
        # recall = 2/3
        # f = (1+2**2)*1*(2/3)/(4*1+(2/3))
        gold = ['entity1', 'entity2', 'entity3']
        pred = ['entity1', 'entity3']
        self.assertEqual(score_fn(gold, pred, 2.0), ((1+2**2)*1*(2/3))/(4*1+(2/3)))
    
    def test_input_no_text(self):
        gold = [1, 2, 3]
        pred = [3, 2]
        self.assertEqual(score_fn(gold, pred, 1.0), 0.8)
    
class TestErrorFunction(unittest.TestCase):

    def test_empty_lists(self):
        self.assertEqual(error_by_category([], []), 0.0)

    def test_empty_predict(self):
        gold = ['entity1', 'entity2', 'entity3']
        pred = []
        self.assertEqual(error_by_category(gold, pred), 1.0)

    def test_empty_gold_answer(self):
        gold = []
        pred = ['entity1', 'entity2', 'entity3']
        self.assertEqual(error_by_category(gold, pred), 0.0)

    def test_return_value(self):
        gold = ['entity1', 'entity2', 'entity3']
        pred = ['entity1', 'entity6', 'entity3']
        self.assertEqual(error_by_category(gold, pred), 1/3)

    def test_return_value_0(self):
        gold = ['entity1', 'entity2', 'entity3']
        pred = gold
        self.assertEqual(error_by_category(gold, pred), 0.0)


    def test_return_value_1(self):
        gold = ['entity1', 'entity2', 'entity3']
        pred = ['entity4', 'entity5', 'entity7']
        self.assertEqual(error_by_category(gold, pred), 1.0)

    def test_return_value_2(self):
        gold = ['entity1', 'entity2', 'entity3']
        pred = ['entity4', 'entity5', 'entity7','entity4', 'entity5', 'entity7']
        self.assertEqual(error_by_category(gold, pred), 2.0)

    def test_input_no_text(self):
        gold = [1, 2, 3]
        pred = [3, 2]
        self.assertEqual(error_by_category(gold, pred), 0.0)


if __name__ == '__main__':
    unittest.main()