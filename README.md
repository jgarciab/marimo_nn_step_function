# Neural Networks: Fitting a Step

Interactive Marimo WASM app for teaching neural-network depth, width, training
time, and overfitting. Fit a network to a fixed step function and compare
training and validation error as the fitted curve changes across epochs.

## Browser app

**[Open the app](https://javier.science/marimo_nn_step_function/)**
GitHub Actions checks the app, builds the WASM bundle, and deploys it to GitHub
Pages on every push to `main`. It runs entirely in the browser, with no server
or external data.

## Run locally

```bash
uv sync
sh run.sh
```

## Explore

Choose depth, width, training sample size, noise SD, epochs, Adam learning rate,
activation, and the step centre (0 or 1). Inspect the sample in the left sidebar,
directly below **New sample**, before clicking **Fit network** in the sidebar
or main panel. Both buttons fit the same network. The results show
training/validation MSE and four fitted curves in one row. Titles contain only
the epoch number. The snapshots match practical 1.9: epochs 10, 24, 250, and the
chosen training limit. Runs of 250 epochs or fewer use four log-spaced epochs.
Changing settings waits for another click before fitting.
**New sample** draws another training sample.

Defaults match section 1.9: one ReLU hidden layer, 1,024 neurons, 50 observations,
noise SD 0.5, 30,000 epochs, and Adam learning rate 0.005. The initial 50-point
sample is the practical's exact R `set.seed(2026)` sample. Its x values and
standard-normal noise are embedded, so no R runtime is required. Changing the
noise or step centre transforms those same observations; **New sample** and
other sample sizes use NumPy draws.

The true mean is 1 between centre ± 0.3 and 0 elsewhere; x is uniform over
centre ± 1, with Gaussian outcome noise. Moving the centre shifts both the
step and the sampling range. Validation uses 500 independent observations,
fixed across training samples and architecture changes. The dashed error
reference is the noise variance. This is empirical validation error; section
1.9 instead calculates expected test error from the known mean.

Full-batch Adam uses Glorot-uniform first/output weights, identity weights
between hidden layers, zero initial biases, and a linear output. Adam places
epsilon in the denominator before second-moment bias correction, as Keras does.
NumPy and TensorFlow generate different initial weights even with the same seed
number, so the trajectories are not identical. The early dip/rise also depends
on the training sample; it need not occur after drawing a new sample.
Errors are recorded at up to 400 log-spaced epochs plus the four snapshots. All requested
epochs are trained. Large networks and long runs take longer to fit.

## Backend

- **NumPy** implements the forward pass, backpropagation, and full-batch Adam.
- **Matplotlib** draws the sample, error curves, and fitted curves as PNG images.
- **marimo** provides the reactive interface and the two fit buttons.
- **Pyodide/WebAssembly** runs Python in the browser export.

The app does not use TensorFlow, Keras, or PyTorch.

## Files

- `app.py`: the standalone marimo app, including data generation and training
- `pyproject.toml` and `uv.lock`: dependencies for local use and export
- `verify_app.py`: numerical and display checks
- `.github/workflows/deploy.yml`: checks, WASM export, and GitHub Pages deployment
- `build/`: optional local WASM export; ignored by git

## Check

```bash
uv run python verify_app.py
uv run python -m marimo check app.py
```

## Export to WASM locally

```bash
bash export_wasm.sh
python -m http.server --directory build
```

The first browser load requires internet to download Marimo and Pyodide.
