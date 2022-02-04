# Estimation of superspreading dispersion

This repository contains a simplified version of the code used in
https://www.nature.com/articles/s41598-021-03126-w

The code makes maximum likelihood estimates of the basic reproduction number *R0*
and dispersion parameter *k* of simulated data (a simple simulation code is also included).

## Running the code
```bash
python simulation.py
python estimate_r0_k.py
```

## Description
The Bayesian log likelihood model is defined in `model.py`.
The script `estimate_r0_k.py` is a simple wrapper
that looks for maximum likelihood estimates for scalar `R0` and `k`.

For more details, see the paper.
