# Neural Network Priors for Bayesian Inverse Problems

## Overview

The repository contains implementation of the function space-based priors, so-called Bayesian neural network (BNN) priors. These priors use neural network structure as an approximation to the unknown function. Weights and biases of such networks are drown from some distributions rather than fixed, that is why these networks are also reffered to as _probabilistic_ neural networks. By changing the distribution on the network weights and biases, it is possible to alter the behaviour of the corresponding BNN prior. For example, if neural network parameters are distributed according to Gaussian distribution, the realisations from the prior are smooth; whereas, if network parameters are drawn from the heavy-tailed Cauchy distribution, the realisations exhibit larger jumps (piecewise constant behavior). 

Codes are suplementary to Chapter 4 of Angelina Senchukova's doctoral thesis titled _"Flexible priors for rough feature reconstruction in Bayesian inversion”_. 

## Installation 

Once the Python-3.8 virtual environment is created, all the required dependencies can be installed by running:
```shell
pip3 install -r requirements.txt
```

## Functionality description 

### Generating realizations from BNN priors 

Currently, the code support one- and two-dimensional BNN priors with network parameters drawn from either Gaussian (smooth) or Cauchy (piecewise constant) distribution.
Realisations from BNN priors can be visualized by running: 
```shell
python3 generate_prior.py -d <distribution> -di <input_dimension>
```
Options for `<distribution>` are `gaussian` or `cauchy`. Options for `<input_dimension>` are `1` (one-dimensional realisations, corresponding to signals) or `2` (two-dimensional realisations, corresponding to images).

Note: to check the comand line arguments and available choice for them, you can use `-h` option when calling the programm:
```shell
python3 generate_prior.py -h
```

For example, after running 
```shell
python3 generate_prior.py -d gaussian -di 1
```
the figures showing different one-dimensional realisations of Gaussian BNN priors (for various numbers of hidden layers and hidden units) will be saved to the directory named `./figures/priors/`. 


