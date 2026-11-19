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
                        value=1024, label="Neurons per layer", show_value=True)
    activation = mo.ui.dropdown(["ReLU", "tanh", "sigmoid", "linear"],
                                value="ReLU", label="Activation")
    epochs = mo.ui.slider(steps=[100, 250, 750, 1500, 3000, 10000, 30000],
                         value=30000, label="Epochs", show_value=True)
    learning_rate = mo.ui.dropdown([0.0001, 0.001, 0.005, 0.01, 0.05, 0.1],
                                   value=0.005, label="Learning rate (Adam)")
    fit_button = mo.ui.run_button(label="Fit network", kind="success", full_width=True)
    main_fit_button = mo.ui.run_button(label="Fit network", kind="success")
    return (activation, centre, depth, epochs, fit_button, learning_rate,
            main_fit_button, new_sample, noise, sample_size, width)


@app.cell
def main_action(main_fit_button):
    main_fit_button
    return


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
    ], width="360px")
    return


@app.cell
def data_helpers(np):
    # Section 1.9's first sample: R set.seed(2026); runif(50, -1, 1); rnorm(50).
    # Store x and standard-normal noise so the noise and centre controls still work.
    reference_sample = np.array([
        [0.397346941754222, 1.90808383464199],
        [0.113061011768878, 1.73094452723732],
        [-0.719720084685832, 0.0581458372592962],
        [-0.428553387522697, 0.645328161775171],
        [0.110738019924611, 1.7256289865063],
        [-0.949737699236721, -0.528966917456246],
        [-0.0675388900563121, 0.166392025612551],
        [0.722021370194852, -0.254723574763946],
        [-0.494997642468661, 0.332782359437909],
        [0.161612674593925, 0.182432812827635],
        [-0.988133486360312, 1.164593749518],
        [0.38373187975958, 0.593492850727868],
        [-0.53774990607053, -0.891779729217369],
        [0.697064924519509, 0.577253310357403],
        [-0.69232951849699, -0.824905581058043],
        [-0.286381370387971, -1.15725385762344],
        [0.090409008320421, 0.777998774006213],
        [-0.99760477244854, -1.20422178597307],
        [-0.36435102019459, 0.30671410214768],
        [-0.96542129246518, -0.833658540642056],
        [-0.316883948165923, 1.41814116350375],
        [-0.313122400082648, 0.711786117850718],
        [-0.534874613396823, -0.402497722862373],
        [-0.877262344583869, 0.799362218714509],
        [-0.175247496459633, 0.426484435556948],
        [-0.628975645639002, -1.16989866993562],
        [-0.178655872587115, -0.206200160293019],
        [-0.719838288612664, -0.930610309243612],
        [-0.989130966365337, 0.449397565791737],
        [0.20347763132304, -0.644806454562313],
        [0.82202025456354, -0.231422689810958],
        [-0.649274432100356, -1.23636801144537],
        [0.462374826427549, -0.960955298223015],
        [-0.967685580719262, 0.133956281569561],
        [0.172231432516128, -0.999052722969283],
        [-0.148363143205643, -0.141470690813213],
        [-0.579018044751137, 0.167329899648813],
        [-0.0455997833050787, -0.198788612671681],
        [0.5096465498209, -0.291207188092097],
        [-0.106341440230608, -1.73439703424929],
        [-0.257596395909786, -0.272728331732604],
        [0.749556809198111, -1.79992948169682],
        [-0.130805385299027, 1.15274097680304],
        [-0.119726446922868, -1.003319485592],
        [-0.836086333263665, 0.148210044292539],
        [-0.930761393159628, 0.519496680749176],
        [0.857303963508457, 0.00543629447576128],
        [-0.41086740558967, 1.34702083465394],
        [0.0384485749527812, -0.847033417295996],
        [-0.527207150124013, 0.443398017315772]
    ])

    def true_mean(x, centre=0):
        return ((x >= centre - 0.3) & (x < centre + 0.3)).astype(float)

    def make_sample(n, noise_sd, centre, seed):
        if n == 50 and seed == 2026:
            x = reference_sample[:, :1].copy() + centre
            y = true_mean(x, centre) + noise_sd * reference_sample[:, 1:2]
            return x, y
        # Separate draws keep x fixed when only the noise level changes.
        rng = np.random.default_rng(seed)
        x = rng.uniform(-1, 1, (n, 1)) + centre
        y = true_mean(x, centre) + rng.normal(0, noise_sd, (n, 1))
        return x, y

    return make_sample, reference_sample, true_mean


@app.cell
def sample(centre, make_sample, new_sample, noise, np, sample_size, true_mean):
    x_train, y_train = make_sample(sample_size.value, noise.value, centre.value,
                                   2026 + new_sample.value)
    # Independent validation observations, never used in the gradient updates.
    x_validation, y_validation = make_sample(500, noise.value, centre.value, 90210)
    x_grid = (np.arange(-100, 101) / 100).reshape(-1, 1) + centre.value
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
def network_helpers(np):
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
        # Use section 1.9's epochs, without interpreting the shape of the curve.
        snapshots = ([10, 24, 250, epochs] if epochs > 250 else
                     np.unique(np.rint(np.geomspace(1, epochs, 4))).astype(int).tolist())
        # Up to 400 log-spaced measurements keep wide networks responsive.
        checkpoints = set(np.unique(np.rint(np.geomspace(1, epochs, 400))).astype(int))
        checkpoints.update(snapshots)
        history, curves = [], {}
        for epoch in range(1, epochs + 1):
            dw, db = gradients(x, y, weights, biases, kind)
            for p, grad, m, v in zip(params, dw + db, moments, velocities):
                m *= 0.9
                m += 0.1 * grad
                v *= 0.999
                v += 0.001 * grad ** 2
                # Keras Adam uses epsilon with the uncorrected second moment.
                step_size = learning_rate * np.sqrt(1 - 0.999 ** epoch) / (1 - 0.9 ** epoch)
                p -= step_size * m / (np.sqrt(v) + 1e-7)
            if epoch in checkpoints:
                train_mse = np.mean((forward(x, weights, biases, kind)[0][-1] - y) ** 2)
                val_mse = np.mean((forward(xv, weights, biases, kind)[0][-1] - yv) ** 2)
                if not np.isfinite([train_mse, val_mse]).all():
                    raise ValueError("Training diverged. Try a smaller learning rate.")
                history.append((epoch, train_mse, val_mse))
            if epoch in snapshots:
                curves[epoch] = forward(grid, weights, biases, kind)[0][-1].ravel().copy()
            if progress is not None and (epoch % 100 == 0 or epoch == epochs):
                progress(epoch, epochs)
        history = np.asarray(history)
        return {"history": history, "curves": curves,
                "parameters": sum(p.size for p in params),
                "weights": weights, "biases": biases}

    return activate, forward, gradients, initialize, train_network


@app.cell
def fit(activation, depth, epochs, fit_button, learning_rate, main_fit_button, mo, train_network,
        width, x_grid, x_train, x_validation, y_train, y_validation):
    mo.stop(not (fit_button.value or main_fit_button.value),
            mo.md("*Inspect the sample, then click Fit network.*"))
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
    _fig, _ax = plt.subplots(figsize=(8.5, 3), layout="constrained")
    _ax.plot(_history[:, 0], _history[:, 1], color="#2B6795", label="Training")
    _ax.plot(_history[:, 0], _history[:, 2], color="#B54708", label="Validation")
    _ax.axhline(noise.value ** 2, color="#777777", linestyle="--", linewidth=1,
                label="Noise variance")
    for _epoch in result["curves"]:
        _ax.axvline(_epoch, color="#aaaaaa", linestyle=":", linewidth=0.8)
    _ax.set(xscale="log", xlabel="Epoch (log scale)", ylabel="Mean squared error", ylim=(0, None))
    _ax.legend(frameon=False, ncol=3, fontsize=9)

    _fits, _axes = plt.subplots(1, 4, figsize=(13, 3.2), sharex=True, sharey=True,
                                layout="constrained")
    for _ax, (_epoch, _prediction) in zip(_axes.flat, result["curves"].items()):
        _ax.scatter(x_train.ravel(), y_train.ravel(), color="#999999", alpha=0.65, s=14)
        _ax.step(x_grid.ravel(), truth.ravel(), where="post", color="#333333",
                 linestyle="--", linewidth=1.2)
        _ax.plot(x_grid.ravel(), _prediction, color="#B54708", linewidth=1.6)
        _ax.set(title=f"Epoch {_epoch:,}",
                xlim=(x_grid.min(), x_grid.max()))
        _ax.title.set_fontsize(11)
    _fits.supxlabel("Predictor x")
    _fits.supylabel("Predicted outcome")

    _results = mo.vstack([
        mo.md("### Follow the error"),
        mo.md(f"**{result['parameters']:,} parameters** · "
              f"Training MSE: **{_history[-1, 1]:.3f}** · "
              f"Validation MSE: **{_history[-1, 2]:.3f}**"),
        plot_image(_fig, "Training and validation mean squared error by epoch"),
        mo.md("Dotted lines mark the epochs plotted below. The dashed line is the "
              "noise variance; error on a finite sample can fall below it."),
        mo.md("### Watch the fitted curve"), plot_image(_fits, "Fitted curves at four epochs"),
        mo.md("Grey = training points · Orange = fitted curve · Dashed = true mean. "
              "Panels follow epoch order. "
              "The starting 50-point sample matches practical 1.9. "
              "This app shows empirical validation MSE; the practical instead "
              "evaluates expected test error from the known true mean. "
              "\n\n**Try it:** use linear activation, add layers, or train for longer. "
              "When does a better training fit stop improving validation error?"),
    ])
    plt.close(_fig)
    plt.close(_fits)
    _results
    return


if __name__ == "__main__":
    app.run()
