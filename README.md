# MCR benchmark workflow on MacroModel testset

## Clone the repo

```
git clone --recursive https://github.com/TheorChemGroup/mcr-benchmark.git
```

## Testset

We use MacroModel testset for our benchmark (see [original paper](https://doi.org/10.1021/ci5001696)). However, there are issues with the structures provided in SI of the original paper (the main concern is the placement of hydrogen atoms). Thus, we implemented a separate workflow for preparation of test macrocycles [in a separate repository](https://gitlab.com/knvvv/macromodel-testset). Moreover, it is more reasonable to start conformational searches from random conformers (not the experimental ones), thus, a separate part of [the preparation workflow](https://gitlab.com/knvvv/macromodel-testset) is dedicated to randomizing conformations of test macrocycles. As a result, this workflow generates 150 structures in sdf-format that are stored in a `project_data/testsets` directory.

## Prepare environments

Create three conda/mamba environments: mcrtest (main env for benchmark), rdkit2024 (contains the newest RDKit for better ETKDG), rdkitref (older RDKit for reference). Most of the packages can be installed via `conda env create -f envname.yaml -n envname`. Whereas, pipeline framework `pysquared` has to be installed via

```
cd pysquared
pip install .
```

IK-driven conformer generator `Ringo` is available [open-source](https://github.com/TheorChemGroup/ringo) and can be install via `pip install ringo-ik`. However, MCR sampling itself is not open-source, so one needs to implement the `run_confsearch` function that uses Ringo's API to perform random sampling as described in the paper and stores the result in a `Confpool` object (MCR is called in the `./scripts/pipelines/benchmark_run.py` file).

Also, make sure that all executables referenced in `./scripts/execscripts/*.sh` exist (actual paths can be modified).

## Run the benchmark

```
mamba activate mcrtest
cd scripts
python main.py
# or (since it is going to take a few days)
nohup python main.py &
```

## Contact information

* Nikolai V. Krivoshchapov – nikolai.krivoshchapov@gmail.com
* Michael G. Medvedev – medvedev.m.g@gmail.com

## License

This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
