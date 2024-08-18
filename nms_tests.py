import unittest

from nms import non_max_suppression


class TestNonMaxSuppression(unittest.TestCase):
    def setUp(self):
        self.t1_boxes = [
            [1, 1, .5, .45, .4, .5],
            [1, .8, .5, .5, .2, .4],
            [1, .7, .25, .35, .3, .1],
            [1, .05, .1, .1, .1, .1],
        ]

        self.c1_boxes = [[1, 1, .5, .45, .4, .5],
                         [1, .7, .25, .35, .3, .1]]

        self.t2_boxes = [
            [1, 1, .5, .45, .4, .5],
            [2, .9, .5, .5, .2, .4],
            [1, .8, .25, .35, .3, .1],
            [1, .05, .1, .1, .1, .1],
        ]

        self.c2_boxes = [
            [1, 1, .5, .45, .4, .5],
            [2, .9, .5, .5, .2, .4],
            [1, .8, .25, .35, .3, .1],
        ]

        self.t3_boxes = [
            [1, .9, .5, .45, .4, .5],
            [1, 1, .5, .5, .2, .4],
            [2, .8, .25, .35, .3, .1],
            [1, .05, .1, .1, .1, .1],
        ]

        self.c3_boxes = [[1, 1, .5, .5, .2, .4],
                         [2, .8, .25, .35, .3, .1]]

        self.t4_boxes = [
            [1, .9, .5, .45, .4, .5],
            [1, 1, .5, .5, .2, .4],
            [1, .8, .25, .35, .3, .1],
            [1, .05, .1, .1, .1, .1],
        ]

        self.c4_boxes = [
            [1, .9, .5, .45, .4, .5],
            [1, 1, .5, .5, .2, .4],
            [1, .8, .25, .35, .3, .1],
        ]

    def test_remove_on_iou(self):
        bboxes = non_max_suppression(
            self.t1_boxes,
            prob_thresh=.2,
            iou_thresh=7 / 20,
            box_format='midpoint',
        )
        self.assertTrue(sorted(bboxes) == sorted(self.c1_boxes))

    def test_keep_on_class(self):
        bboxes = non_max_suppression(
            self.t2_boxes,
            prob_thresh=.2,
            iou_thresh=7 / 20,
            box_format='midpoint',
        )
        self.assertTrue(sorted(bboxes) == sorted(self.c2_boxes))

    def test_remove_on_iou_and_class(self):
        bboxes = non_max_suppression(
            self.t3_boxes,
            prob_thresh=.2,
            iou_thresh=7 / 20,
            box_format='midpoint',
        )
        self.assertTrue(sorted(bboxes) == sorted(self.c3_boxes))

    def test_keep_on_iou(self):
        bboxes = non_max_suppression(
            self.t4_boxes,
            prob_thresh=.2,
            iou_thresh=9 / 20,
            box_format='midpoint',
        )
        self.assertTrue(sorted(bboxes) == sorted(self.c4_boxes))


if __name__ == '__main__':
    print('Running Non Max Suppression Tests:')
    unittest.main()
