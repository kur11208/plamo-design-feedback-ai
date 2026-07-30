from __future__ import annotations

import unittest

from security_config import read_env_flag


class SecurityConfigTest(unittest.TestCase):
    def test_local_only_features_are_disabled_without_explicit_opt_in(self) -> None:
        self.assertFalse(read_env_flag("PLAMO_ENABLE_IMAGE_UPLOAD", environ={}))
        self.assertFalse(read_env_flag("PLAMO_ENABLE_LOCAL_LLM", environ={}))

    def test_local_only_features_require_known_true_value(self) -> None:
        self.assertTrue(
            read_env_flag(
                "PLAMO_ENABLE_IMAGE_UPLOAD",
                environ={"PLAMO_ENABLE_IMAGE_UPLOAD": "true"},
            )
        )
        self.assertFalse(
            read_env_flag(
                "PLAMO_ENABLE_IMAGE_UPLOAD",
                environ={"PLAMO_ENABLE_IMAGE_UPLOAD": "enabled"},
            )
        )


if __name__ == "__main__":
    unittest.main()
