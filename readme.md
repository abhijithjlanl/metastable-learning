# Metastable Learning

This has been tested on Julia 1.9.2.

Code to recreate the numerical experiments in the paper is in the `Experiments`
folder. `LS_env` contains the Julia environment with the packages needed to run
the files.

## Julia environment

From the repository root, start Julia with the project environment active:

```bash
julia --project=LS_env
```

The first time you use the repository, instantiate the environment from the
Julia REPL:

```julia
using Pkg
Pkg.instantiate()
```

You can also activate and instantiate the environment from a normal Julia
session:

```julia
using Pkg
Pkg.activate("LS_env")
Pkg.instantiate()
```

For one-off scripts, pass the environment on the command line:

```bash
julia --project=LS_env path/to/script.jl
```

## Curie-Weiss experiments

The main Curie-Weiss experiments are in `Experiments/CW_tests`. The scripts in
this folder use relative paths, so run them from inside that directory:

```bash
cd Experiments/CW_tests
```

Collect samples and learn the Curie-Weiss model parameters:

```bash
julia --project=../../LS_env cw_datacollect.jl
```

This writes or updates `data_cw.jld2`. The default settings use large sample
counts and can take a long time; edit the parameter lists near the top of
`cw_datacollect.jl` for smaller test runs.

Create the summary plots from `data_cw.jld2`:

```bash
julia --project=../../LS_env cw_plotter.jl
```

This writes `error_cw.pdf`, `prob_cw.pdf`, and `mag_cw.pdf`.

Create the free-energy histogram plot:

```bash
julia --project=../../LS_env free_energy_plotter.jl
```

This writes `FE_histogram.pdf`. If the cached sample file named in the script is
not present, the script will generate new Glauber samples, which can be slow.

Run the sampler test script:

```bash
julia --project=../../LS_env cw_sampler_test.jl
```

This script also uses a large default sample count. Reduce `M_list` in the file
for a quick smoke test.

The MLE and pseudolikelihood contour plots are in
`Experiments/CW_tests/MLE_test`:

```bash
cd Experiments/CW_tests/MLE_test
julia --project=../../../LS_env test_MLE.jl
```

`test_MLE.jl` expects `samples_MCMC` to be available. To run it as a standalone
script, uncomment the sampling line that defines `samples_MCMC`, or load samples
before evaluating the plotting section. It writes `PLE_loss.pdf`,
`MLE_loss.pdf`, and `PLE_losszoom.pdf`.

## Spin Glass experiments

Spin Glass experiment documentation is still TODO.
