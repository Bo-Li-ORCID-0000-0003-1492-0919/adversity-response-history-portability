"""Presentation checks for the aggregate Extended Data figures."""
from pathlib import Path
import csv
import importlib.util
import unittest
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('extended_renderer',ROOT/'src/figures/reproduce_extended_data_figures.py')
M=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)

class ExtendedFigureLabelTests(unittest.TestCase):
    def test_event_styles_are_distinguishable(self):
        styles=[M.FAMILY_STYLES[k] for k in M.FAMILIES]
        self.assertEqual(len(set(c for c,_ in styles)),4)
        self.assertEqual(len(set(s for _,s in styles)),4)

    def test_no_raw_variable_labels_in_reference_figures(self):
        forbidden=('persistence_2y_z','persistence_4y_z','ancova_level','current_state','omega_total',
                   'financial_strain','between_history_sd','most_recent','residualized_mean',
                   'overlap_allowed','IPOW_truncated_1_99','complete_four_points','-0.000')
        for path in (ROOT/'expected_outputs/extended_data_figures').glob('*.pdf'):
            text=' '.join(page.extract_text() or '' for page in PdfReader(path).pages)
            for token in forbidden:
                with self.subTest(file=path.name,token=token):self.assertNotIn(token,text)

    def test_reference_widths_are_180_mm(self):
        for path in (ROOT/'expected_outputs/extended_data_figures').glob('*.pdf'):
            page=PdfReader(path).pages[0]
            self.assertAlmostEqual(float(page.mediabox.width),180/25.4*72,places=3)
            self.assertLessEqual(float(page.mediabox.height),700)

    def test_forests_have_complete_display_mappings(self):
        for i in (3,4,6,7):
            path=ROOT/f'source_data/extended_data/Extended_Data_Figure_{i}_source.csv'
            with path.open(encoding='utf-8-sig',newline='') as f:
                for row in csv.DictReader(f):
                    label=M.forest_label(row['label'])
                    self.assertNotIn('_',label)
                    self.assertTrue(label.strip())

    def test_weighting_counts_are_episode_counts(self):
        path=ROOT/'source_data/extended_data/Extended_Data_Figure_8_source.csv'
        with path.open(encoding='utf-8-sig',newline='') as f:
            rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),48)
        self.assertTrue(all(row['source_n_column']=='n_events' for row in rows))
        text=PdfReader(ROOT/'expected_outputs/extended_data_figures/Extended_Data_Figure_8.pdf').pages[0].extract_text()
        self.assertIn('n episodes',text)

if __name__=='__main__':unittest.main()
