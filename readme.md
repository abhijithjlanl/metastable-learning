# Metastable Learning

This has been tested on Julia 1.9.2. The checked-in manifest was resolved with
Julia 1.9.3, so use Julia 1.9.x for the most reliable reproduction.

Code to recreate the numerical experiments in the paper is in the `Experiments`
folder. The root `Project.toml` and `Manifest.toml` contain the Julia
environment with the packages needed to run the files.

## Julia environment

From the repository root, start Julia with the project environment active:

```bash
julia --project=.
```

The first time you use the repository, instantiate the environment from the
Julia REPL:

```julia
using Pkg
Pkg.instantiate()
```

If you are already in a Julia session from the repository root, activate the
environment directly:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

For one-off scripts, pass the environment on the command line:

```bash
julia --project=. path/to/script.jl
```

## Curie-Weiss experiments

The main Curie-Weiss experiments are in `Experiments/CW_tests`. The scripts in
this folder use relative paths, so run them from inside that directory:

```bash
cd Experiments/CW_tests
```

Collect samples and learn the Curie-Weiss model parameters:

```bash
julia --project=../.. cw_datacollect.jl
```

This writes or updates `data_cw.jld2`. The defaults match the plotting scripts:
`n_spins = 5000`, `J * n_spins = 1.2`, `h = 0.04`, Glauber sampling, and
`M = 2^22, 2^24, ..., 2^32`. These are large runs; edit the parameter lists near
the top of `cw_datacollect.jl` for smaller smoke tests.

Create the summary plots from `data_cw.jld2`:

```bash
julia --project=../.. cw_plotter.jl
```

This writes `error_cw.pdf`, `prob_cw.pdf`, and `mag_cw.pdf`.

Create the free-energy histogram plot:

```bash
julia --project=../.. free_energy_plotter.jl
```

This writes `FE_histogram.pdf`. If the cached sample file named in the script is
not present, the script will generate new Glauber samples, which can be slow.

Run the sampler test script:

```bash
julia --project=../.. cw_sampler_test.jl
```

This script also uses a large default sample count. Reduce `M_list` in the file
for a quick smoke test.

The MLE and pseudolikelihood contour plots are in
`Experiments/CW_tests/MLE_test`. `test_MLE.jl` is not standalone as committed:
it expects `samples_MCMC` to be available. To reproduce these plots, uncomment
the sampling line that defines `samples_MCMC`, or load samples before evaluating
the plotting section. Then run:

```bash
cd Experiments/CW_tests/MLE_test
julia --project=../../.. test_MLE.jl
```

It writes `PLE_loss.pdf`, `MLE_loss.pdf`, and `PLE_losszoom.pdf`.

## Spin-glass learning experiments

Spin-glass learning experiments are in
`Experiments/PottsSGLearning/n12_beta_sweep`. These scripts use `@__DIR__` to
find the repository paths, so they can be run from the repository root.

To reproduce the checked-in plots from the checked-in result files:

```bash
julia --project=. Experiments/PottsSGLearning/n12_beta_sweep/plotter.jl
julia --project=. Experiments/PottsSGLearning/n12_beta_sweep/plotter_energy.jl
julia --project=. Experiments/PottsSGLearning/n12_beta_sweep/plotter_panelb.jl
```

The scripts write:

- `n12_24_beta_sweep.pdf` from `results.jld2` and `results_n24.jld2`
- `n12_24_energy_sweep.pdf` from `results.jld2` and `results_n24.jld2`
- `n12_24_panelb.pdf` from `results_panelb.jld2`

To regenerate the result files before plotting, run:

```bash
julia --project=. Experiments/PottsSGLearning/n12_beta_sweep/datacollector.jl
julia --project=. Experiments/PottsSGLearning/n12_beta_sweep/datacollector_n24.jl
julia --project=. Experiments/PottsSGLearning/n12_beta_sweep/datacollector_panelb.jl
```

The Potts utilities used by these experiments live in `PottsPSpin`, and the
Potts pseudolikelihood learner is in
`Experiments/PottsSGLearning/potts_sumoflocal_PLE.jl`.

## License



This code is provided under a BSD license as part of the Optimization, Inference and Learning for Advanced Networks project, C18014.
