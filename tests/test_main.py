from __future__ import annotations

import subprocess
import sys
import unittest


class MainTests(unittest.TestCase):
    def test_version_flag_prints_package_name_and_version(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "enhancer.main", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.stdout.strip(), "ai-video-enhancer 0.2.0")
        self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
