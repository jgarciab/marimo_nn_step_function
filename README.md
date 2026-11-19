# Neural Networks: Fitting a Step

Interactive Marimo WASM app for teaching neural-network depth, width, training
time, and overfitting. Fit a network to a fixed step function and compare
training and validation error as the fitted curve changes across epochs.

## Live app

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
directly below **New sample**, before clicking
**Fit network**. The results show training/validation MSE and fitted curves at
four epochs in one row, ordered chronologically. The panels include the lowest
recorded validation error, a clear dip and subsequent rebound if present, and
the final epoch. Other well-spaced epochs fill missing or overlapping landmarks.
Dip/rebound detection uses a smoothed validation curve and requires a substantial
rise followed by recovery; the minimum uses the original recorded errors.
Changing settings waits for another click before fitting.
**New sample** draws another training sample.

The true mean is 1 between centre ± 0.3 and 0 elsewhere; x is uniform over
centre ± 1, with Gaussian outcome noise. Moving the centre shifts both the
step and the sampling range. Validation uses 500 independent observations,
fixed across training samples and architecture changes. The dashed error
reference is the noise variance. This is empirical validation error; section
1.9 instead calculates expected test error from the known mean.

Full-batch Adam uses Glorot-uniform first/output weights, identity weights
between hidden layers, zero initial biases, and a linear output. Errors are
recorded at up to 400 log-spaced epochs; predictions at these checkpoints allow
the four snapshot epochs to be selected after fitting. All requested
epochs are trained. Large networks and long runs take longer to fit.

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
