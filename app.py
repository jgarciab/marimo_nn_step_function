# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo==0.23.0", "matplotlib>=3.10.8", "numpy>=2.4.4"]
# ///

import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full", app_title="Neural networks: fitting a step")


@app.cell
def imports():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#bbbbbb", "text.color": "#333333",
        "axes.labelcolor": "#333333", "font.size": 11,
        "figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.2,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return mo, np, plt


@app.cell
def title(mo):
    mo.md("""
    # Neural networks: fitting a step

    *Javier Garcia-Bernardo · Utrecht University & ODISSEI SoDa*

    Choose a sample and a network, inspect the points, then **Fit network**.
    How do depth, width and training time change the fit and validation error?
    """)
    return


@app.cell
def controls(mo):
    sample_size = mo.ui.slider(steps=[20, 50, 100, 200, 500], value=50,
                              label="Training sample size", show_value=True)
    noise = mo.ui.slider(0, 1, step=0.1, value=0.5, label="Noise SD", show_value=True)
    centre = mo.ui.dropdown([0, 1], value=0, label="Step centre")
    new_sample = mo.ui.button(value=0, on_click=lambda count: count + 1,
                              label="New sample")
    depth = mo.ui.slider(1, 8, value=1, step=1, label="Hidden layers", show_value=True)
    width = mo.ui.slider(steps=[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
                        value=64, label="Neurons per layer", show_value=True)
    activation = mo.ui.dropdown(["ReLU", "tanh", "sigmoid", "linear"],
                                value="ReLU", label="Activation")
    epochs = mo.ui.slider(steps=[100, 250, 750, 1500, 3000, 10000, 30000],
                         value=3000, label="Epochs", show_value=True)
    learning_rate = mo.ui.dropdown([0.0001, 0.001, 0.005, 0.01, 0.05, 0.1],
                                   value=0.005, label="Learning rate (Adam)")
    fit_button = mo.ui.run_button(label="Fit network", kind="success", full_width=True)
    return (activation, centre, depth, epochs, fit_button, learning_rate,
            new_sample, noise, sample_size, width)


@app.cell
def sidebar(activation, centre, depth, epochs, fit_button, learning_rate, mo,
            new_sample, noise, sample_plot, sample_size, width):
    mo.sidebar([
        mo.md("### Sample"), sample_size, noise, centre, new_sample,
        sample_plot,
        mo.md("Blue = sample · Dashed = true mean"),
        mo.md("### Network"), depth, width, activation, epochs, learning_rate,
        fit_button,
        mo.md("Each epoch uses the full training sample. The output is linear. "
              "Validation uses 500 independent observations."),
    ])
    return


@app.cell
def data_helpers(np):
    def true_mean(x, centre=0):
        return ((x >= centre - 0.3) & (x < centre + 0.3)).astype(float)

    def make_sample(n, noise_sd, centre, seed):
        # Separate draws keep x fixed when only the noise level changes.
        rng = np.random.default_rng(seed)
        x = rng.uniform(-1, 1, (n, 1)) + centre
        y = true_mean(x, centre) + rng.normal(0, noise_sd, (n, 1))
        return x, y

    return make_sample, true_mean


@app.cell
def sample(centre, make_sample, new_sample, noise, np, sample_size, true_mean):
    x_train, y_train = make_sample(sample_size.value, noise.value, centre.value,
                                   2026 + new_sample.value)
    # Independent validation observations, never used in the gradient updates.
    x_validation, y_validation = make_sample(500, noise.value, centre.value, 90210)
    x_grid = np.linspace(-1, 1, 401).reshape(-1, 1) + centre.value
    truth = true_mean(x_grid, centre.value)
    return truth, x_grid, x_train, x_validation, y_train, y_validation


@app.cell
def plot_helper(mo):
    from io import BytesIO

    def plot_image(figure, alt):
        buffer = BytesIO()
        figure.savefig(buffer, format="png", bbox_inches="tight")
        return mo.image(buffer.getvalue(), alt=alt, width="100%")

    return (plot_image,)


@app.cell
def sample_preview(plot_image, plt, truth, x_grid, x_train, y_train):
    _fig, _ax = plt.subplots(figsize=(3.3, 2.5), layout="constrained")
    _ax.scatter(x_train.ravel(), y_train.ravel(), s=12, alpha=0.7,
                color="#2B6795")
    _ax.step(x_grid.ravel(), truth.ravel(), where="post", color="#333333",
             linestyle="--", label="True mean")
    _ax.set(xlabel="Predictor x", ylabel="Outcome y", xlim=(x_grid.min(), x_grid.max()))
    _ax.tick_params(labelsize=8)
    _ax.xaxis.label.set_size(9)
    _ax.yaxis.label.set_size(9)
    sample_plot = plot_image(_fig, "Training sample and true step function")
    plt.close(_fig)
    return (sample_plot,)


@app.cell
def snapshot_helper(np):
    def choose_snapshots(history):
        """Keep the validation minimum, a substantial dip/rebound, and the end."""
        epochs, loss = history[:, 0].astype(int), history[:, 2]
        selected = {}

        def add(index, label):
            epoch = int(epochs[index])
            selected[epoch] = " / ".join(filter(None, [selected.get(epoch), label]))

        add(int(np.argmin(loss)), "Minimum")
        add(len(loss) - 1, "Final")
        # Smooth only for detecting a clear dip followed by a rebound and recovery.
        # The minimum and plotted losses still use the unsmoothed measurements.
        smooth = np.convolve(np.pad(loss, (3, 3), mode="edge"), np.ones(7) / 7, mode="valid")
        threshold = max(0.02 * np.ptp(smooth), 0.03 * np.median(smooth), 1e-8)
        best_pair, best_prominence = None, threshold
        for peak in range(2, len(loss) - 2):
            if not (smooth[peak] >= smooth[peak - 1] and smooth[peak] > smooth[peak + 1]):
                continue
            dip = 1 + int(np.argmin(smooth[1:peak]))
            recovery = np.min(smooth[peak + 1:])
            prominence = min(smooth[peak] - smooth[dip], smooth[peak] - recovery)
            if prominence > best_prominence:
                best_pair, best_prominence = (dip, peak), prominence
        if best_pair is not None:
            for index, label, extreme in zip(best_pair, ["Dip", "Rebound"], [np.argmin, np.argmax]):
                lo, hi = max(1, index - 3), min(len(loss) - 1, index + 4)
                add(lo + int(extreme(loss[lo:hi])), label)
        # Missing or overlapping landmarks are filled with well-spaced epochs.
        log_epochs = np.log(epochs)
        while len(selected) < min(4, len(epochs)):
            distance = np.min(np.abs(log_epochs[:, None] - np.log(list(selected))), axis=1)
            index = int(np.argmax(distance))
            add(index, "Early fit" if index == 0 else "Intermediate fit")
        return dict(sorted(selected.items()))

    return (choose_snapshots,)


@app.cell
def network_helpers(choose_snapshots, np):
    def activate(z, kind):
        if kind == "ReLU":
            return np.maximum(z, 0)
        if kind == "tanh":
            return np.tanh(z)
        if kind == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(z, -700, 700)))
        return z

    def initialize(depth, width, seed=2027):
        rng = np.random.default_rng(seed)
        dims = [1] + [width] * depth + [1]
        weights, biases = [], []
        for i, (fan_in, fan_out) in enumerate(zip(dims[:-1], dims[1:])):
            limit = np.sqrt(6 / (fan_in + fan_out))
            # Match the practical: Glorot at the ends, identity between hidden layers.
            weights.append(np.eye(width) if 0 < i < depth else
                           rng.uniform(-limit, limit, (fan_in, fan_out)))
            biases.append(np.zeros((1, fan_out)))
        return weights, biases

    def forward(x, weights, biases, kind):
        values = [x]
        preactivations = []
        for i, (w, b) in enumerate(zip(weights, biases)):
            z = values[-1] @ w + b
            preactivations.append(z)
            values.append(z if i == len(weights) - 1 else activate(z, kind))
        return values, preactivations

    def gradients(x, y, weights, biases, kind):
        values, zs = forward(x, weights, biases, kind)
        delta = 2 * (values[-1] - y) / len(y)
        dw, db = [None] * len(weights), [None] * len(weights)
        for i in range(len(weights) - 1, -1, -1):
            dw[i] = values[i].T @ delta
            db[i] = delta.sum(axis=0, keepdims=True)
            if i > 0:
                delta = delta @ weights[i].T
                if kind == "ReLU":
                    delta *= zs[i - 1] > 0
                elif kind == "tanh":
                    delta *= 1 - values[i] ** 2
                elif kind == "sigmoid":
                    delta *= values[i] * (1 - values[i])
        return dw, db

    def train_network(x, y, xv, yv, grid, *, depth, width, kind, epochs,
                      learning_rate, progress=None):
        weights, biases = initialize(depth, width)
        params = weights + biases
        moments = [np.zeros_like(p) for p in params]
        velocities = [np.zeros_like(p) for p in params]
        # Up to 400 log-spaced measurements keep wide networks responsive.
        checkpoints = set(np.unique(np.rint(np.geomspace(1, epochs, 400))).astype(int))
        history, recorded_curves = [], {}
        for epoch in range(1, epochs + 1):
            dw, db = gradients(x, y, weights, biases, kind)
            for p, grad, m, v in zip(params, dw + db, moments, velocities):
                m *= 0.9
                m += 0.1 * grad
                v *= 0.999
                v += 0.001 * grad ** 2
                p -= learning_rate * (m / (1 - 0.9 ** epoch)) / (
                    np.sqrt(v / (1 - 0.999 ** epoch)) + 1e-7)
            if epoch in checkpoints:
                train_mse = np.mean((forward(x, weights, biases, kind)[0][-1] - y) ** 2)
                val_mse = np.mean((forward(xv, weights, biases, kind)[0][-1] - yv) ** 2)
                if not np.isfinite([train_mse, val_mse]).all():
                    raise ValueError("Training diverged. Try a smaller learning rate.")
                history.append((epoch, train_mse, val_mse))
                recorded_curves[epoch] = forward(grid, weights, biases, kind)[0][-1].ravel().copy()
            if progress is not None and (epoch % 100 == 0 or epoch == epochs):
                progress(epoch, epochs)
        history = np.asarray(history)
        snapshot_labels = choose_snapshots(history)
        curves = {epoch: recorded_curves[epoch] for epoch in snapshot_labels}
        return {"history": history, "curves": curves, "snapshot_labels": snapshot_labels,
                "parameters": sum(p.size for p in params),
                "weights": weights, "biases": biases}

    return activate, forward, gradients, initialize, train_network


@app.cell
def fit(activation, depth, epochs, fit_button, learning_rate, mo, train_network,
        width, x_grid, x_train, x_validation, y_train, y_validation):
    mo.stop(not fit_button.value, mo.md("*Inspect the sample, then click Fit network.*"))
    with mo.status.progress_bar(total=epochs.value, title="Fitting network",
                                show_rate=False, remove_on_exit=True) as _bar:
        def _progress(epoch, total):
            _bar.update(increment=epoch % 100 or 100, subtitle=f"Epoch {epoch:,} / {total:,}")

        try:
            result = train_network(
                x_train, y_train, x_validation, y_validation, x_grid,
                depth=depth.value, width=width.value, kind=activation.value,
                epochs=epochs.value, learning_rate=learning_rate.value, progress=_progress,
            )
        except ValueError as _error:
            mo.stop(True, mo.md(str(_error)))
    return (result,)


@app.cell
def results(mo, noise, np, plot_image, plt, result, truth, x_grid, x_train, y_train):
    _history = result["history"]
    _best = _history[np.argmin(_history[:, 2])]
    _fig, _ax = plt.subplots(figsize=(8.5, 3), layout="constrained")
    _ax.plot(_history[:, 0], _history[:, 1], color="#2B6795", label="Training")
    _ax.plot(_history[:, 0], _history[:, 2], color="#B54708", label="Validation")
    _ax.axhline(noise.value ** 2, color="#777777", linestyle="--", linewidth=1,
                label="Noise variance")
    for _epoch in result["curves"]:
        _ax.axvline(_epoch, color="#aaaaaa", linestyle=":", linewidth=0.8)
    _ax.scatter(_best[0], _best[2], color="#B54708", marker="*", s=90, zorder=4)
    _ax.set(xscale="log", xlabel="Epoch (log scale)", ylabel="Mean squared error", ylim=(0, None))
    _ax.legend(frameon=False, ncol=3, fontsize=9)

    _fits, _axes = plt.subplots(1, 4, figsize=(13, 3.2), sharex=True, sharey=True,
                                layout="constrained")
    for _ax, (_epoch, _prediction) in zip(_axes.flat, result["curves"].items()):
        _ax.scatter(x_train.ravel(), y_train.ravel(), color="#999999", alpha=0.65, s=14)
        _ax.step(x_grid.ravel(), truth.ravel(), where="post", color="#333333",
                 linestyle="--", linewidth=1.2)
        _ax.plot(x_grid.ravel(), _prediction, color="#B54708", linewidth=1.6)
        _ax.set(title=f"Epoch {_epoch:,}\n{result['snapshot_labels'][_epoch]}",
                xlim=(x_grid.min(), x_grid.max()))
        _ax.title.set_fontsize(11)
    _fits.supxlabel("Predictor x")
    _fits.supylabel("Predicted outcome")

    _results = mo.vstack([
        mo.md("### Follow the error"),
        mo.md(f"**{result['parameters']:,} parameters** · "
              f"Final training MSE: **{_history[-1, 1]:.3f}** · "
              f"Final validation MSE: **{_history[-1, 2]:.3f}**"),
        plot_image(_fig, "Training and validation mean squared error by epoch"),
        mo.md(f"★ Lowest recorded validation error: **{_best[2]:.3f}** "
              f"at epoch **{int(_best[0]):,}**. Dotted lines mark the fits below. "
              "The dashed line is the noise variance; error on a finite sample can fall below it."),
        mo.md("### Watch the fitted curve"), plot_image(_fits, "Fitted curves at four epochs"),
        mo.md("Grey = training points · Orange = fitted curve · Dashed = true mean. "
              "Panels follow epoch order. If there is no clear dip and rebound, "
              "the extra panels show other stages of training. "
              "\n\n**Try it:** use linear activation, add layers, or train for longer. "
              "When does a better training fit stop improving validation error?"),
    ])
    plt.close(_fig)
    plt.close(_fits)
    _results
    return


if __name__ == "__main__":
    app.run()
