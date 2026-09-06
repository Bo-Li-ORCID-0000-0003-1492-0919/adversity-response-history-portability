from pathlib import Path
import importlib.util
import json
import unittest

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


def load_validator():
    path = ROOT / "scripts/validate_public_release.py"
    spec = importlib.util.spec_from_file_location("validate_public_release", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PublicReleaseTests(unittest.TestCase):
    def test_expected_display_counts(self):
        self.assertEqual(len(list((ROOT / "expected_outputs/main_figures").glob("Figure_*.pdf"))), 4)
        self.assertEqual(len(list((ROOT / "expected_outputs/extended_data_figures").glob("Extended_Data_Figure_*.pdf"))), 8)

    def test_expected_displays_are_one_page(self):
        for path in (ROOT / "expected_outputs").rglob("*.pdf"):
            self.assertEqual(len(PdfReader(path).pages), 1)


    def test_display_registry_sources_exist(self):
        registry = json.loads((ROOT / "config/display_registry.json").read_text(encoding="utf-8"))
        for section in ("figures", "tables"):
            for key, record in registry[section].items():
                with self.subTest(section=section, key=key):
                    self.assertTrue((ROOT / record["source"]).is_file())

    def test_public_tables_present(self):
        self.assertEqual(len(list((ROOT / "source_data/tables").glob("Table_*.xlsx"))), 3)
        self.assertEqual(len(list((ROOT / "extended_data_tables").glob("Extended_Data_Table_*.xlsx"))), 2)
        self.assertEqual(len(list((ROOT / "supplementary_tables").glob("Supplementary_Table_*.xlsx"))), 4)

    def test_public_count_suppression(self):
        validator = load_validator()
        validator.check_small_counts()
        validator.check_cross_file_sample_counts()


if __name__ == "__main__":
    unittest.main()
