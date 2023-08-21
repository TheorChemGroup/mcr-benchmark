# Ringo testing system on MacroModel testset

## How to reproduce our testing?

### Prepare testset

See 

```
python create_small_testset.py
python create_testset.py
```

### Prepare environments

Double-check that all `exec_*` scripts contain valid paths to CREST, XTB and SCHRODINGER directories.

### Execute conformational searches

Record isomorphisms counts:

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

