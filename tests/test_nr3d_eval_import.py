import importlib.util
import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "nr3d_eval.py"


class Nr3dEvalImportTest(unittest.TestCase):
    def test_nr3d_eval_module_imports(self) -> None:
        spec = importlib.util.spec_from_file_location("nr3d_eval_module", SCRIPT_PATH)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()
