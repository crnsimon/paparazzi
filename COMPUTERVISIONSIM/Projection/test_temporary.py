import unittest
from projection_functions import VideoFeed

class TestVideoFeed(unittest.TestCase):
    def setUp(self):
        self.video_feed = VideoFeed(r'C:\Users\Jonathan van Zyl\Documents\GitHub\paparazzi\Data_gitignore\AE4317_2019_datasets\cyberzoo_poles_panels\20190121-140205')
        self.video_feed.frame_files = ['60483805.jpg', '60483806.jpg', '60483807.jpg']
        self.video_feed.index = 1

    def test_find_time(self):
        result = self.video_feed.find_time()
        #self.assertEqual(result, 60.483805)
        self.assertEqual(result, 60.582822)

if __name__ == '__main__':
    unittest.main()