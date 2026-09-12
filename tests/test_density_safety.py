"""Synthetic-data tests; no licensed observations or fitted outputs are used."""
import ast
import math
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_function(filename, name, extra=None):
    tree = ast.parse((ROOT / "src/analysis" / filename).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    ns = {"np": np, "math": math, "BOOT_SUCCESS": 500, "SEED": 1}
    ns.update(extra or {})
    module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    exec(compile(module, str(filename), "exec"), ns)
    return ns[name]


class DensitySafetyTests(unittest.TestCase):
    def setUp(self):
        self.old = load_function("run_nmh_confirmatory_rebuild.py", "metrics")
        self.new = load_function("run_nmh_measurement_adjudication.py", "metric_values")
        self.y = np.array([0.0, 2.0, 4.0])
        self.pred = np.array([0.5, 1.5, 3.0])

    def test_positive_training_variance_keeps_gaussian_formula(self):
        variance = 1.7
        expected = np.mean(-.5 * (np.log(2*np.pi*variance) + (self.y-self.pred)**2 / variance))
        for result in [self.old(self.y, self.pred, variance=variance), self.new(self.y, self.pred, variance)]:
            self.assertAlmostEqual(result["mean_log_predictive_density"], expected, places=14)
            self.assertAlmostEqual(result["RMSE"], np.sqrt(np.mean((self.y-self.pred)**2)), places=14)

    def test_invalid_scalar_variance_is_never_replaced_by_test_mse(self):
        for variance in [None, 0.0, -1.0, np.nan, np.inf, -np.inf]:
            with self.subTest(variance=variance):
                with self.assertRaisesRegex(ValueError, "evaluation-set MSE"):
                    self.old(self.y, self.pred, variance=variance)
                with self.assertRaisesRegex(ValueError, "evaluation-set MSE"):
                    self.new(self.y, self.pred, variance)

    def test_invalid_row_variance_is_never_replaced(self):
        for variance in [0, -1, np.nan, np.inf]:
            with self.assertRaisesRegex(ValueError, "evaluation-set MSE"):
                self.new(self.y, self.pred, np.array([1.0, variance, 2.0]))

    def test_row_specific_training_variances_and_weights(self):
        v = np.array([.5, 1.5, 2.5]); w = np.array([2., 1., 3.])
        expected = np.average(-.5*(np.log(2*np.pi*v)+(self.y-self.pred)**2/v), weights=w)
        self.assertAlmostEqual(self.new(self.y, self.pred, v, w)["mean_log_predictive_density"], expected, places=14)

    def test_descriptive_bootstrap_singular_retries_terminate(self):
        def singular(*args):
            return np.array(["synthetic"]), np.zeros((1, 2, 2)), np.zeros((1, 2))
        fn = load_function("run_nmh_confirmatory_rebuild.py", "bootstrap_fixed_effects", {"person_gls_contributions": singular})
        with self.assertRaisesRegex(RuntimeError, "required successful"):
            fn(None, None, None, None, success_target=2, seed=1)


if __name__ == "__main__":
    unittest.main()
