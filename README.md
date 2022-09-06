# Evaluation of Volume Representation Networks for Meteorological Ensemble Compression

Kevin Höhlein, Sebastian Weiss and Rüdiger Westermann

![Teaser](analysis/figures/title_image.png)

## Abstract

Recent studies have shown that volume scene representation networks constitute powerful means to transform 3D scalar fields into extremely compact representations, from which the initial field samples can be randomly accessed. 
In this work, we evaluate the capabilities of such networks to compress meteorological ensemble data, which comprise many single weather forecast simulations. 
We analyze whether these networks can effectively exploit similarities between the ensemble members, and how alternative classical compression approaches perform in comparison. 
Since meteorological ensembles contain different physical parameters with various statistical characteristics and variations on multiple scales of magnitude, we analyze the impact of data normalization schemes on learning quality. 
Along with an evaluation of the trade-offs between reconstruction quality and network model parameterization, we compare compression ratios and reconstruction quality for different model architectures and alternative compression schemes. 

This repository contains the code and settings to reproduce the results of the paper.


## Requirements

 - NVIDIA GPU with RTX, e.g. RTX20xx or RTX30xx (we use an RTX2070)
 - CUDA 11
 - OpenGL with GLFW and GLM
 - Python 3.8 or higher, see `environment.yml` for the required packages

Tested system:

- Ubuntu 20.04, gcc 9.3.0, CUDA 11.1, Python 3.8, PyTorch 1.8

## Installation

Training and analysis codes are written in pure Python and rely on fast CUDA implementations from the [pyrenderer](https://github.com/shamanDevel/fV-SRN) project by Sebastian Weiss. 
The project further depends on an installation of the referenced implementations of floating-point compressors [SZ3](https://github.com/szcompressor/SZ3) and [TThresh](https://github.com/rballester/tthresh.git).
A Python environment can be set up using `environment.yml`. For setting up the code for reproduction, run 

    bash setup.sh 

For details (and common pitfalls) concerning the installation procedures, we refer to the respective repositories.

## Project structure

## How to train

## How to reproduce the figures

## Comparisons against other compression algorithms
