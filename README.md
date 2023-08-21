# Ringo benchmark workflow on MacroModel testset

## How to reproduce our testing?

### Prepare testset

We use MacroModel testset for our benchmark (see [original paper](https://doi.org/10.1021/ci5001696)). However, there are issues with the structures provided in SI of the original paper (the main concern is the placement of hydrogen atoms). Thus, we implemented a separate workflow for preparation of test macrocycles [in a separate repository](https://gitlab.com/knvvv/macromodel-testset). Moreover, it is more reasonable to start conformational searches from random conformers (not the experimental ones), thus, a separate part of [the preparation workflow](https://gitlab.com/knvvv/macromodel-testset) is dedicated to randomizing conformations of test macrocycles. As a result, this workflow generates 150 structures in sdf-format that are stored in a `start_conformers` directory.

As soon as starting conformers are generated, json-files of testing data can be created (note that the path to `start_conformers` directory has to be set in both scripts):

```
python create_small_testset.py # Generates 'small_testcases.json'
python create_testset.py # Generates 'testcases.json'
```

### Prepare environments

Double-check that all `exec_*` scripts contain valid paths to CREST, XTB and SCHRODINGER directories.

Scripts of this workflow use three conda environments and two python variants (usual interpreter and Intel Python). Two envs for usual interpreter can be reproduced from YAML files:

```
conda env create -f gnu_env.yml -n full
conda env create -f rpython_env.yml -n rpyenv
```

For better performance we sometimes use intelpython, however, this environment cannot be automatically reproduced from yaml-file (`intel_env.yml`). If it is desired, one can create this env by hand to have all the key packages of `full` env (pandas, rdkit, etc.) 

Additionally, all two (or three including intel) environments need to have `pyxyz` and `ringo` installed. To install them, follow README files at corresponding repositories of [pyxyz](https://gitlab.com/knvvv/pyxyz) and [ringo](https://gitlab.com/knvvv/ringo) - these libraries can either be built from source or downloaded as prebuilt binaries.

When all two (three) environments have been created, add commands for their activation in the `environments.py` script (the corresponding entry has to be uncommented if intelpython is used).

### Execute conformational searches

Record isomorphism counts for more efficient RMSD filtering:

```
python calc_niso.py
```

Execute conformational searches with all methods:

```
python run_ringo.py
python run_rdkit.py
python run_mtd.py
python run_crest.py # Must be executed twice: DO_GENCROSSING_FAILED=False and DO_GENCROSSING_FAILED=True
python run_mmbasic.py
python run_mmring.py
```

### Optimize and post-process ensembles

```
python optimize.py
```

### Perform diversity analysis and generate full PDF

```
python diversity_analysis.py
```

### Plot timing benchmarks and generate CSVs with testset overview and timings

Open `analyze_timings.ipynb` and execute all the cells.

## Repository layout

