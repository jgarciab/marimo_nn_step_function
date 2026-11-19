"""Numerical and display checks: uv run python verify_app.py."""

import unittest
import base64
from io import BytesIO
import re

import numpy as np
from PIL import Image

from app import app


class StepChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.outputs, cls.defs = app.run()

    def test_preview_does_not_train(self):
        self.assertNotIn("result", self.defs)
        self.assertEqual(self.defs["x_train"].shape, (50, 1))
        self.assertEqual(self.defs["x_validation"].shape, (500, 1))

    def test_preview_contains_a_plot_and_fit_button(self):
        html = "".join(getattr(output, "text", "") for output in self.outputs)
        sidebar = html.split("<marimo-sidebar", 1)[1].split("</marimo-sidebar>", 1)[0]
        self.assertIn("data:image/png;base64,", sidebar)
        self.assertLess(sidebar.index("New sample"), sidebar.index("data:image/png;base64,"))
        self.assertLess(sidebar.index("data:image/png;base64,"), sidebar.index("Network"))
        self.assertIn("Fit network", html)

    def test_snapshots_find_dip_rebound_and_minimum_in_epoch_order(self):
        epochs = np.arange(1, 101)
        loss = np.interp(epochs, [1, 15, 35, 70, 100], [0.6, 0.25, 0.45, 0.15, 0.25])
        history = np.c_[epochs, loss, loss]
        labels = self.defs["choose_snapshots"](history)
        self.assertEqual(list(labels), [15, 35, 70, 100])
        self.assertEqual(list(labels.values()), ["Dip", "Rebound", "Minimum", "Final"])

    def test_monotone_and_noisy_curves_do_not_invent_dips(self):
        epochs = np.arange(1, 101)
        for loss in [np.linspace(1, 0.2, 100), np.full(100, 0.3),
                     np.linspace(0.4, 0.2, 100) + 0.0005 * np.sin(epochs)]:
            labels = self.defs["choose_snapshots"](np.c_[epochs, loss, loss])
            self.assertEqual(len(labels), 4)
            self.assertEqual(list(labels), sorted(labels))
            self.assertIn(100, labels)
            self.assertIn(int(epochs[np.argmin(loss)]), labels)
            self.assertFalse(any("Dip" in label or "Rebound" in label for label in labels.values()))

    def test_shift_preserves_problem(self):
        make = self.defs["make_sample"]
        x0, y0 = make(50, 0.5, 0, 2026)
        x1, y1 = make(50, 0.5, 1, 2026)
        np.testing.assert_allclose(x1, x0 + 1)
        np.testing.assert_array_equal(y0, y1)
        for centre in [0, 1]:
            x, y = make(500, 0, centre, 42)
            expected = ((x >= centre - 0.3) & (x < centre + 0.3)).astype(float)
            np.testing.assert_array_equal(y, expected)

    def test_gradients_against_finite_differences(self):
        forward = self.defs["forward"]
        rng = np.random.default_rng(20)
        x, y = rng.normal(size=(7, 1)), rng.normal(size=(7, 1))
        for kind in ["linear", "ReLU", "tanh", "sigmoid"]:
            w, b = self.defs["initialize"](2, 3)
            # Avoid ReLU kinks for numerical differentiation.
            for bias in b:
                bias += rng.normal(0, 0.2, size=bias.shape)
            dw, db = self.defs["gradients"](x, y, w, b, kind)
            for param, grad in zip(w + b, dw + db):
                for index in np.ndindex(param.shape):
                    original = param[index]
                    param[index] = original + 1e-6
                    plus = np.mean((forward(x, w, b, kind)[0][-1] - y) ** 2)
                    param[index] = original - 1e-6
                    minus = np.mean((forward(x, w, b, kind)[0][-1] - y) ** 2)
                    param[index] = original
                    self.assertAlmostEqual(grad[index], (plus - minus) / 2e-6, places=6)

    def test_training_snapshots_and_validation_isolation(self):
        d = self.defs
        args = (d["x_train"], d["y_train"], d["x_validation"])
        settings = dict(depth=2, width=8, kind="ReLU", epochs=300, learning_rate=0.005)
        result = d["train_network"](*args, d["y_validation"], d["x_grid"], **settings)
        changed = d["train_network"](*args, d["y_validation"] + 10, d["x_grid"], **settings)
        self.assertEqual(len(result["curves"]), 4)
        self.assertEqual(list(result["curves"]), sorted(result["curves"]))
        self.assertIn(int(result["history"][np.argmin(result["history"][:, 2]), 0]), result["curves"])
        self.assertIn(300, result["curves"])
        self.assertEqual(result["parameters"], 3 * 8 + 1 + 8 ** 2 + 8)
        self.assertTrue(np.isfinite(result["history"]).all())
        self.assertLess(result["history"][-1, 1], result["history"][0, 1])
        for a, b in zip(result["weights"], changed["weights"]):
            np.testing.assert_array_equal(a, b)
        prediction = d["forward"](d["x_grid"], result["weights"], result["biases"], "ReLU")[0][-1]
        np.testing.assert_array_equal(result["curves"][300], prediction.ravel())
        train_prediction = d["forward"](d["x_train"], result["weights"], result["biases"], "ReLU")[0][-1]
        self.assertAlmostEqual(result["history"][-1, 1], np.mean((train_prediction - d["y_train"]) ** 2))

    def test_linear_network_matches_least_squares(self):
        d = self.defs
        result = d["train_network"](d["x_train"], d["y_train"], d["x_validation"],
            d["y_validation"], d["x_grid"], depth=3, width=4, kind="linear",
            epochs=1500, learning_rate=0.005)
        design = np.c_[np.ones(50), d["x_train"]]
        coef = np.linalg.lstsq(design, d["y_train"], rcond=None)[0]
        expected = np.c_[np.ones(len(d["x_grid"])), d["x_grid"]] @ coef
        np.testing.assert_allclose(result["curves"][1500], expected.ravel(), atol=1e-5)

    def test_results_cells_execute(self):
        d = self.defs
        controls = ["activation", "centre", "depth", "epochs", "fit_button",
                    "learning_rate", "new_sample", "noise", "sample_size", "width"]
        # Simulate the button press and execute the actual training/display cells.
        d["fit_button"]._value = True
        try:
            outputs, fitted_defs = app.run(defs={name: d[name] for name in controls})
        finally:
            d["fit_button"]._value = False
        self.assertIn("result", fitted_defs)
        self.assertEqual(fitted_defs["result"]["history"][-1, 0], 3000)
        html = "".join(getattr(output, "text", "") for output in outputs)
        self.assertIn("Follow the error", html)
        self.assertIn("Watch the fitted curve", html)
        self.assertEqual(html.count("data:image/png;base64,"), 3)
        images = re.findall(r"data:image/png;base64,([A-Za-z0-9+/=]+)", html)
        fits = Image.open(BytesIO(base64.b64decode(images[-1])))
        self.assertGreater(fits.width / fits.height, 3)


if __name__ == "__main__":
    unittest.main()
