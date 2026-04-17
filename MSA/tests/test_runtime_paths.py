import os
import unittest

import config


class RuntimePathTests(unittest.TestCase):
    def test_cmumosi_label_path_resolves_to_existing_file(self):
        self.assertTrue(os.path.exists(config.PATH_TO_LABEL["CMUMOSI"]))

    def test_get_save_dir_returns_saved_subdir(self):
        model_dir = config.get_save_dir("model")
        self.assertTrue(model_dir.endswith(os.path.join("saved", "model")))

