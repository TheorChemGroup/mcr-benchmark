import os
import json
import shutil
from typing import Callable, Optional

from pysquared import Transform
import pysquared.transforms.transform_templates as templates

from chemscripts import geom as ccgeom
from chemscripts.utils import assert_availability

from utils import confsearch

#
# PERFORMS THE ACTUAL BENCHMARK RUNS & BASIC ENSEMBLE POST-PROCESSING
#

VERY_LARGE_ENSEMBLES_METHODS = ('ringo-vs-crest', 'ringoCN-vs-crest', 'ringo-vs-mmring')

# For RMSD filtering duplicate conformers
PYXYZ_SETTINGS = {
    'rmsd_cutoff': 0.2, # A
    'mirror_match': True,
    'print_status': False,

    # Threshold may be enabled for faster RMSD filtering of very large ensembles
    'energy_threshold': 1.0, # kcal/mol
}

RDKIT2022_ACTIVATION = """\
source /s/ls4/users/knvvv/mambaactivate
mamba activate rdkitref
"""
RDKIT2024_ACTIVATION = """\
source /s/ls4/users/knvvv/mambaactivate
mamba activate rdkit2024
"""

# For filtering low energy conformers
# ENERGY_THRESHOLD = 15.0 # kcal/mol


def benchmark_dataitems() -> dict[str, dict]:
    return { # CHECK Uncomment everything important
        # For additional verification of starting geometries
        'random_testmols_correct': {'type': 'object', 'keys': ['testset', 'testcase']},

        # For injection of earlier (without pipelines) computed ensembles
        'old_sampling_cpuload_json': {'type': 'file', 'mask': './benchmark_runs/old_ensembles/{testset}/cpuload_{old_method}.json'},
        'old_sampling_cpuload_avg': {'type': 'object', 'keys': ['testset', 'testcase', 'old_method']},
        'old_sampling_singlethread_cpuload': {'type': 'object', 'keys': ['testset', 'testcase', 'old_method']},
        'old_sampling_cpuload_total': {'type': 'object', 'keys': ['testset', 'testcase', 'old_method']},
        
        'old_sampling_csv': {'type': 'file', 'mask': './benchmark_runs/old_ensembles/{testset}/{old_method}_df.csv'},
        'old_sampling_xyz': {'type': 'file', 'mask': './benchmark_runs/old_ensembles/{testset}/{old_method}_conformers/{testcase}_{timelimit}.xyz'},
        'old_sampling_df': {'type': 'object', 'keys': ['testset', 'old_method']},
        'old_sampling_entries': {'type': 'object', 'keys': ['testset', 'testcase', 'old_method']},
        'old_sampling_entries_fixed': {'type': 'object', 'keys': ['testset', 'testcase', 'old_method', 'timelimit']},

        # Ensembles & stats for pure confsearch (timelimit=600 only)
        'mcr_ensemble_xyz': {'type': 'file', 'mask': './benchmark_runs/gen_ensembles/mcr_{timelimit}_spec/{testset}_{testcase}.xyz'},
        'mcr_stats_single': {'type': 'file', 'mask': './benchmark_runs/gen_ensembles/mcr_{timelimit}_spec/{testset}_{testcase}.json'},
        'rdkit_ensemble_xyz': {'type': 'file', 'mask': './benchmark_runs/gen_ensembles/rdkit_{timelimit}_spec/{testset}_{testcase}.xyz'},
        'rdkit_stats_single': {'type': 'file', 'mask': './benchmark_runs/gen_ensembles/rdkit_{timelimit}_spec/{testset}_{testcase}.json'},
        'mtd_ensemble_xyz': {'type': 'file', 'mask': './benchmark_runs/gen_ensembles/mtd_{timelimit}_spec/{testset}_{testcase}.xyz'},
        'mtd_stats_single': {'type': 'file', 'mask': './benchmark_runs/gen_ensembles/mtd_{timelimit}_spec/{testset}_{testcase}.json'},
        'old_ensemble_xyz': {'type': 'file', 'mask': './benchmark_runs/gen_ensembles/old_{timelimit}_spec/{method}_{testset}_{testcase}.xyz'},
        'old_stats_single': {'type': 'file', 'mask': './benchmark_runs/gen_ensembles/old_{timelimit}_spec/{method}_{testset}_{testcase}.json'},
        'heavy_old_ensemble_xyz': {'type': 'file', 'mask': './benchmark_runs/gen_ensembles/heavyold_{timelimit}_spec/{method}_{testset}_{testcase}.xyz'},
        'heavy_old_stats_single': {'type': 'file', 'mask': './benchmark_runs/gen_ensembles/heavyold_{timelimit}_spec/{method}_{testset}_{testcase}.json'},
        'rdkitv1_ensemble_xyz': {'type': 'file', 'mask': './benchmark_runs/gen_ensembles/rdkitv1_{timelimit}_spec/{testset}_{testcase}.xyz'},
        'rdkitv1_stats_single': {'type': 'file', 'mask': './benchmark_runs/gen_ensembles/rdkitv1_{timelimit}_spec/{testset}_{testcase}.json'},
        'rdkitv3_2024_ensemble_xyz': {'type': 'file', 'mask': './benchmark_runs/gen_ensembles/rdkitv3-2024_{timelimit}_spec/{testset}_{testcase}.xyz'},
        'rdkitv3_2024_stats_single': {'type': 'file', 'mask': './benchmark_runs/gen_ensembles/rdkitv3-2024_{timelimit}_spec/{testset}_{testcase}.json'},

        # To run MCR with termination conditions referencing CREST's, we need to extract testcases used in CREST into separate file
        'randomized_heavy_testmols_sdfs': {'type': 'file', 'mask': './testsets/{testset}/heavy_testmols_start/{testcase}.sdf'},
        'heavy_final_summary_object': {'type': 'object', 'keys': ['testset', 'testcase']},

        # Ensembles & stats for MCR with some weird termination conditions for in-depth comparison with some particular method
        'mcr_vs_rdkit_ensemble_xyz': {'type': 'file', 'mask': './benchmark_runs/gen_ensembles/mcr-vs-rdkit_{timelimit}_spec/{testset}_{testcase}.xyz'},
        'mcr_vs_rdkit_stats_single': {'type': 'file', 'mask': './benchmark_runs/gen_ensembles/mcr-vs-rdkit_{timelimit}_spec/{testset}_{testcase}.json'},
        'mcr_vs_rdkit2024_ensemble_xyz': {'type': 'file', 'mask': './benchmark_runs/gen_ensembles/mcr-vs-rdkit2024_{timelimit}_spec/{testset}_{testcase}.xyz'},
        'mcr_vs_rdkit2024_stats_single': {'type': 'file', 'mask': './benchmark_runs/gen_ensembles/mcr-vs-rdkit2024_{timelimit}_spec/{testset}_{testcase}.json'},
        'mcr_vs_mtd_ensemble_xyz': {'type': 'file', 'mask': './benchmark_runs/gen_ensembles/mcr-vs-mtd_{timelimit}_spec/{testset}_{testcase}.xyz'},
        'mcr_vs_mtd_stats_single': {'type': 'file', 'mask': './benchmark_runs/gen_ensembles/mcr-vs-mtd_{timelimit}_spec/{testset}_{testcase}.json'},
        'mcr_vs_crest_ensemble_xyz': {'type': 'file', 'mask': './benchmark_runs/gen_ensembles/mcr-vs-crest_{timelimit}_spec/{testset}_{testcase}.xyz'},
        'mcr_vs_crest_stats_single': {'type': 'file', 'mask': './benchmark_runs/gen_ensembles/mcr-vs-crest_{timelimit}_spec/{testset}_{testcase}.json'},
        'mcr_vs_mmring_ensemble_xyz': {'type': 'file', 'mask': './benchmark_runs/gen_ensembles/mcr-vs-mmring_{timelimit}_spec/{testset}_{testcase}.xyz'},
        'mcr_vs_mmring_stats_single': {'type': 'file', 'mask': './benchmark_runs/gen_ensembles/mcr-vs-mmring_{timelimit}_spec/{testset}_{testcase}.json'},

        # Repeated MCR sampling but with CONSTRAIN_AMIDE_BONDS enabled
        'mcrCN_ensemble_xyz': {'type': 'file', 'mask': './benchmark_runs/gen_ensembles/mcrCN_{timelimit}_spec/{testset}_{testcase}.xyz'},
        'mcrCN_stats_single': {'type': 'file', 'mask': './benchmark_runs/gen_ensembles/mcrCN_{timelimit}_spec/{testset}_{testcase}.json'},
        'mcrCN_vs_rdkit_ensemble_xyz': {'type': 'file', 'mask': './benchmark_runs/gen_ensembles/mcrCN-vs-rdkit_{timelimit}_spec/{testset}_{testcase}.xyz'},
        'mcrCN_vs_rdkit_stats_single': {'type': 'file', 'mask': './benchmark_runs/gen_ensembles/mcrCN-vs-rdkit_{timelimit}_spec/{testset}_{testcase}.json'},
        'mcrCN_vs_mtd_ensemble_xyz': {'type': 'file', 'mask': './benchmark_runs/gen_ensembles/mcrCN-vs-mtd_{timelimit}_spec/{testset}_{testcase}.xyz'},
        'mcrCN_vs_mtd_stats_single': {'type': 'file', 'mask': './benchmark_runs/gen_ensembles/mcrCN-vs-mtd_{timelimit}_spec/{testset}_{testcase}.json'},
        'mcrCN_vs_crest_ensemble_xyz': {'type': 'file', 'mask': './benchmark_runs/gen_ensembles/mcrCN-vs-crest_{timelimit}_spec/{testset}_{testcase}.xyz'},
        'mcrCN_vs_crest_stats_single': {'type': 'file', 'mask': './benchmark_runs/gen_ensembles/mcrCN-vs-crest_{timelimit}_spec/{testset}_{testcase}.json'},

        # Ensembles & stats merged into single FileItems
        'raw_ensemble_xyz': {'type': 'file', 'mask': './benchmark_runs/raw_ensembles/{method}_{timelimit}/{testset}_{testcase}.xyz'},
        'csrun_stats_single': {'type': 'file', 'mask': './benchmark_runs/raw_ensembles/{method}_{timelimit}/{testset}_{testcase}.json'},
        
        'nonzero_summary_object': {'type': 'object', 'keys': ['testset', 'testcase', 'method', 'timelimit']},
        'nonzero_testmols_sdf': {'type': 'file', 'mask': './benchmark_runs/nonzero_testmols/{testset}/{method}_{timelimit}/{testcase}.sdf'},

        'fixed_ordering_ensemble_xyz': {'type': 'file', 'mask': './benchmark_runs/reorderedatoms_ensembles/{method}_{timelimit}/{testset}_{testcase}.xyz'},
        'orderfix_nonzero_testmols_sdf': {'type': 'file', 'mask': './benchmark_runs/orderfix_nonzero_testmols/{testset}/{method}_{timelimit}/{testcase}.sdf'},

        # Various optimizations
        'mmff_opt_ensemble_xyz': {'type': 'file', 'mask': './benchmark_runs/opt_temp/mmff/{method}_{timelimit}/{testset}_{testcase}.xyz'},
        'gfnff_opt_ensemble_xyz': {'type': 'file', 'mask': './benchmark_runs/opt_temp/gfnff/{method}_{timelimit}/{testset}_{testcase}.xyz'},
        'gfnff_postopt_ensemble_xyz': {'type': 'file', 'mask': './benchmark_runs/opt_temp/gfnff_post/{method}_{timelimit}/{testset}_{testcase}.xyz'},
        'opt_ensemble_xyz': {'type': 'file', 'mask': './benchmark_runs/postprocessed/{opttype}/optimized_ensembles/{method}_{timelimit}/{testset}_{testcase}.xyz'},

        # Postprocessing after optimization
        'postopt_nonzero_testmols_sdf': {'type': 'file', 'mask': './benchmark_runs/postopt_nonzero_testmols/{testset}/{method}_{timelimit}/{testcase}.sdf'},
        'filtered_ensemble_xyz': {'type': 'file', 'mask': './benchmark_runs/postprocessed/{opttype}/filtered_ensembles/{method}_{timelimit}/{testset}_{testcase}.xyz'},
    }


def ringo_benchmark():
    import sys
    import os
    import ntpath
    import json
    module_dir = INSERT_HERE
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    os.chdir(os.path.dirname(__file__))

    import ringo
    from utils.confsearch import Timings, TimingContext, get_amide_bonds, sdf_to_confpool
    from utils.custom_sampling import SamplingCustomizer
    from chemscripts import geom as ccgeom
    from icecream import install
    install()

    RINGO_METHOD = INSERT_HERE
    CONSTRAIN_AMIDE_BONDS = INSERT_HERE
    MODES = INSERT_HERE

    start_sdf = sys.argv[1]
    num_automorphisms = int(sys.argv[2])
    molname = ntpath.basename(start_sdf)
    ic(start_sdf, num_automorphisms, molname)

    for timelimit, mode_settings in MODES.items():
        ringo.cleanup()
        timer = Timings()
        p = ringo.Confpool()
        start_mol = ccgeom.Molecule(sdf=start_sdf)

        init_kwargs = {}
        if CONSTRAIN_AMIDE_BONDS:
            sampling_customizer = SamplingCustomizer(start_sdf, sampling_rules={
                "amide_treat": {
                    "allowed_configurations": [0.0, 180.0],
                    "sampling_width": 20.0,
                    "filtering_width": 20.0,
                    "mandatory": False,
                },
                "special_dihedrals": []
            })
            fixed_dihedrals = sampling_customizer.get_fixed_bonds()
            if len(fixed_dihedrals) > 0:
                init_kwargs['request_free'] = fixed_dihedrals

        mol = ringo.Molecule(sdf=start_sdf, **init_kwargs)
        if CONSTRAIN_AMIDE_BONDS:
            sampling_customizer.set_sampling_limits(mol)

        rmsd_settings = None
        if num_automorphisms > 500:
            biggest_frag_atoms = mol.get_biggest_ringfrag_atoms()
            rmsd_settings = {
                'isomorphisms': {
                    'ignore_elements': [node for node in start_mol.G.nodes if node not in biggest_frag_atoms],
                },
                'rmsd': {
                    'threshold': 0.2,
                    'mirror_match': True,
                }
            }
        else:
            rmsd_settings = 'default'
        assert rmsd_settings is not None
        # ringo.cleanup()

        postopt_settings= [
            # Disable both 1st and 2nd potential
            {'enabled': False},
            {'enabled': False},
        ]

        termination_conditions = {
            key: value
            for key, value in mode_settings['termination'].items()
            if key in ('max_conformers', 'max_tries', 'timelimit')
        }
        if 'max_conformers' not in termination_conditions and 'timelimit' in termination_conditions:
            if termination_conditions['timelimit'] <= 600:
                termination_conditions['max_conformers'] = 10000
            else:
                termination_conditions['max_conformers'] = -1


        def save_stats(nconf, nunique, time):
            stats = {
                'method': RINGO_METHOD,
                'testcase': molname,
                'nconf': nconf,
                'nunique': nunique,
                'time': time,
            }

            with open(mode_settings['stats'], 'w') as f:
                json.dump(stats, f)

        if termination_conditions['max_conformers'] == 0:
            save_stats(nconf=0, nunique=0, time=0.0)
            continue

        if isinstance(termination_conditions['timelimit'], float):
            termination_conditions['timelimit'] = int(termination_conditions['timelimit'])

        # raise Exception(f"Running '{start_sdf}' with term conditions '{MODES}', particularly '{termination_conditions}'")
        with TimingContext(timer, molname):
            results = ringo.run_confsearch(
                mol,
                pool=p,
                rmsd_settings=rmsd_settings,
                postopt_settings=postopt_settings,
                geometry_validation = {
                    "ringo": {
                        "bondlength": 0.05,
                        "valence": 3.0,
                        "dihedral": 3.0
                    }
                },
                **termination_conditions
            )
        
        ntotal = results['nsucc']

        if len(p) > 0:
            p.save_xyz(mode_settings['ensemble'])

        save_stats(nconf=ntotal, nunique=len(p), time=timer.time_elapsed)


def rdkit_benchmark():
    import sys
    import os
    import ntpath
    import json
    import time

    module_dir = INSERT_HERE
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    os.chdir(os.path.dirname(__file__))

    import ringo
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import numpy as np
    from utils.confsearch import Timings, TimingContext
    from chemscripts import geom as ccgeom

    from icecream import install
    install()

    RDKIT_METHOD = INSERT_HERE
    MODES = INSERT_HERE
    SMILES = INSERT_HERE

    start_sdf = sys.argv[1]
    molname = ntpath.basename(start_sdf)
    ic(start_sdf, molname, SMILES)

    params = Chem.SmilesParserParams()
    params.removeHs = False
    params.sanitize = True
    mol = Chem.MolFromSmiles(SMILES, params)

    for timelimit, filenames in MODES.items():
        p = ringo.Confpool()
        timer = Timings()

        params = Chem.SmilesParserParams()
        params.removeHs = False
        params.sanitize = True
        mol = Chem.MolFromSmiles(SMILES, params)
        
        # Generate conformers continuously for 'timelimit' seconds
        conformer_idx = 0
        start_time = time.time()
        with TimingContext(timer, molname):
            while (time.time() - start_time) < timelimit:
                # Embed molecule with random coordinates
                AllChem.EmbedMolecule(mol)

                try:
                    conformer = mol.GetConformer()
                except:
                    continue

                # Write molecule as XYZ file
                geom = np.zeros((mol.GetNumAtoms(), 3))
                for i in range(mol.GetNumAtoms()):
                    pos = conformer.GetAtomPosition(i)
                    geom[i, 0] = pos.x
                    geom[i, 1] = pos.y
                    geom[i, 2] = pos.z
                p.include_from_xyz(geom, f"Conformer {conformer_idx}")
                conformer_idx += 1
        p.atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

        if len(p) > 1:
            p.generate_connectivity(0, mult=1.3, ignore_elements=['HCarbon'])
            p.generate_isomorphisms()
            ntotal = len(p)
            p.rmsd_filter(0.2, mirror_match=True, print_status=False)
        else:
            ntotal = 0
        
        if len(p) > 0:
            p.save_xyz(filenames['ensemble'])

        stats = {
            'method': RDKIT_METHOD,
            'testcase': molname,
            'nconf': ntotal,
            'nunique': len(p),
            'time': timer.time_elapsed,
        }
        with open(filenames['stats'], 'w') as f:
            json.dump(stats, f)


def mtd_benchmark():
    import sys
    import os
    import ntpath
    import json
    import time

    module_dir = INSERT_HERE
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    temp_dir = os.path.dirname(__file__)
    os.chdir(temp_dir)

    from chemscripts.utils import add_directory_to_path
    EXECSCRIPTS_DIR = INSERT_HERE
    add_directory_to_path(EXECSCRIPTS_DIR)

    import ringo
    import numpy as np
    from utils.confsearch import Timings, TimingContext, gen_mtd
    from chemscripts import geom as ccgeom

    from icecream import install
    install()

    MTD_METHOD = INSERT_HERE
    MODES = INSERT_HERE

    start_sdf = sys.argv[1]
    molname = ntpath.basename(start_sdf)
    ic(start_sdf, molname)

    for timelimit, filenames in MODES.items():
        p = ringo.Confpool()
        start_mol = ccgeom.Molecule(sdf=start_sdf)
        timer = Timings()
        
        with TimingContext(timer, molname):
            gen_mtd(
                temp_dir=temp_dir,
                sdf_name=start_sdf,
                p=p,
                timelimit=timelimit,
                charge=start_mol.total_charge(),
            )

        if len(p) > 1:
            p.generate_connectivity(0, mult=1.3, ignore_elements=['HCarbon'])
            p.generate_isomorphisms()
            ntotal = len(p)
            p.rmsd_filter(0.2, mirror_match=True, print_status=False)
        else:
            ntotal = 0
        
        if len(p) > 0:
            p.save_xyz(filenames['ensemble'])

        stats = {
            'method': MTD_METHOD,
            'testcase': molname,
            'nconf': ntotal,
            'nunique': len(p),
            'time': timer.time_elapsed,
        }
        with open(filenames['stats'], 'w') as f:
            json.dump(stats, f)


def fix_atom_ordering():
    import sys
    import os
    import ntpath
    import json
    import shutil
    import numpy as np
    import networkx as nx
    module_dir = INSERT_HERE
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    os.chdir(os.path.dirname(__file__))

    import ringo
    from chemscripts import geom as ccgeom
    from utils.confsearch import (
        ConformerInfo
    )
    
    from icecream import install
    install()
    
    METHODS_REQUIRE_REORDERING = INSERT_HERE
    method = INSERT_HERE

    # Unpack the 'input_data' dict
    raw_xyz_path = INSERT_HERE
    assert raw_xyz_path == sys.argv[1]
    initial_sdf_path = INSERT_HERE
    assert initial_sdf_path == sys.argv[2]
    fixed_xyz_path = INSERT_HERE
    assert fixed_xyz_path == sys.argv[3]

    assert os.path.isfile(raw_xyz_path), f"Start XYZ '{raw_xyz_path}' not found"
    assert os.path.isfile(initial_sdf_path), f"Start SDF '{initial_sdf_path}' not found"

    if method not in METHODS_REQUIRE_REORDERING:
        shutil.copy2(raw_xyz_path, fixed_xyz_path)
        sys.exit(0)

    def get_graph_mapping(my_graph: nx.Graph, ref_graph: nx.Graph) -> dict[int, int] | None:
        node_match=lambda n1, n2: n1['symbol'] == n2['symbol']
        good = nx.is_isomorphic(my_graph, ref_graph, node_match=node_match)
        if not good:
            return None
        
        GM = nx.algorithms.isomorphism.GraphMatcher(my_graph, ref_graph, node_match=node_match)
        for isom in GM.isomorphisms_iter():
            return isom

    ccmol = ccgeom.Molecule(sdf=initial_sdf_path)
    required_graph = ccmol.G

    raw_p = ringo.Confpool()
    raw_p.include_from_file(raw_xyz_path)
    for conf_idx in range(len(raw_p)):
        raw_p.generate_connectivity(conf_idx, mult=1.3, sdf_name='xyz_connectivity.sdf')
        raw_graph: nx.Graph = raw_p.get_connectivity()

        correction_map: dict[int, int] | None = get_graph_mapping(required_graph, ref_graph=raw_graph)
        if correction_map is not None:
            break

    assert correction_map is not None

    raw_atom_symbols: list[str] = raw_p.atom_symbols
    reordered_atom_symbols = [raw_atom_symbols[correction_map[i]] for i in range(len(raw_atom_symbols))]
    _, correct_atom_symbols = ccmol.as_xyz()
    assert reordered_atom_symbols == correct_atom_symbols, f"Incorrect topology of {raw_xyz_path} vs. {initial_sdf_path}"

    result_p = ringo.Confpool()
    for m in raw_p:
        raw_xyz: np.ndarray = m.xyz
        result_p.include_from_xyz(
            np.array([raw_xyz[correction_map[i]] for i in range(len(raw_xyz))]),
            m.descr
        )
    result_p.atom_symbols = reordered_atom_symbols
    result_p.save_xyz(fixed_xyz_path)


def mmff_optimize_ensemble():
    import sys
    import os
    import ntpath
    import json
    import shutil

    module_dir = INSERT_HERE
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    
    script_dir = os.path.dirname(__file__)
    os.chdir(script_dir)
    XYZ_TEMP_DIR = os.path.join(script_dir, 'temp_xyzs')
    os.mkdir(XYZ_TEMP_DIR)

    from ringo import Confpool
    from chemscripts import geom as ccgeom
    from utils.confsearch import (
        optimize_mmff_and_geomcheck,
        confpool_to_batches,
        multiproc_mmff_opt,
        ConformerInfo
    )

    from icecream import install
    install()

    start_xyz = INSERT_HERE
    assert start_xyz == sys.argv[1]
    initial_sdf = INSERT_HERE
    assert initial_sdf == sys.argv[2]
    res_xyz = INSERT_HERE
    assert res_xyz == sys.argv[3]
    ic(start_xyz, initial_sdf, res_xyz)

    NUM_PROCS = INSERT_HERE

    assert os.path.isfile(start_xyz), f"Start XYZ named '{start_xyz}' not found"
    assert os.path.isfile(initial_sdf), f"Initial SDF named '{initial_sdf}' not found"

    # CHECK Make sure that such skipping is needed
    if os.path.isfile(res_xyz):
        with open(res_xyz, 'r') as file:
            # Read the first two lines
            first_line = file.readline()
            second_line = file.readline()
        if '"energy":' in second_line:
            sys.exit(0)

    initial_p = Confpool()
    try:
        initial_p.include_from_file(start_xyz)
    except:
        print(f"OPT FAILED on '{start_xyz}'")
        ic(start_xyz, initial_sdf, res_xyz)
        sys.exit(0)
    ConformerInfo.standardize_confpool_descriptions(initial_p)
    
    if NUM_PROCS == 1 or len(initial_p) < 100:
        optimize_mmff_and_geomcheck(
            initial_p=initial_p,
            initial_sdf=initial_sdf,
            res_xyz=res_xyz,
        )
    else:
        assert NUM_PROCS < len(initial_p)
        initial_p_list = confpool_to_batches(initial_p, NUM_PROCS)
        thread_data = []
        outputs_in_order = []
        for i, cur_p in enumerate(initial_p_list):
            input_name = os.path.join(XYZ_TEMP_DIR, f'input_{i}.xyz')
            output_name = os.path.join(XYZ_TEMP_DIR, f'output_{i}.xyz')
            cur_p.save_xyz(input_name)
            thread_data.append([
                input_name,
                initial_sdf,
                output_name
            ])
            outputs_in_order.append(output_name)
        
        multiproc_mmff_opt(thread_data, NUM_PROCS)

        output_p = Confpool()
        for output_name in outputs_in_order:
            output_p.include_from_file(output_name)
        output_p.save_xyz(res_xyz)
        shutil.rmtree(XYZ_TEMP_DIR)


def gfnff_optimize_ensemble():
    import sys
    import os
    import ntpath
    import subprocess
    import shutil
    from typing import Dict, Tuple, Any

    module_dir = INSERT_HERE
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    script_dir = os.path.dirname(__file__)
    os.chdir(script_dir)

    XTB_TEMP_DIR = os.path.join(script_dir, 'xtb_runs')
    os.mkdir(XTB_TEMP_DIR)
    XYZ_TEMP_DIR = os.path.join(script_dir, 'input_xyzs')
    os.mkdir(XYZ_TEMP_DIR)

    from chemscripts.utils import add_directory_to_path, assert_availability

    from ringo import Confpool
    from chemscripts import geom as ccgeom
    from chemscripts.utils import write_xyz
    from utils.confsearch import (
        optimize_gfnff_and_geomcheck,
        multiproc_gfnff_opt,
        confpool_to_batches,
        ConformerInfo
    )

    from icecream import install
    install()

    EXECSCRIPTS_DIR = INSERT_HERE
    add_directory_to_path(EXECSCRIPTS_DIR)
    assert_availability('exec_xtbopt.sh')

    start_xyz = INSERT_HERE
    assert start_xyz == sys.argv[1]
    initial_sdf = INSERT_HERE
    assert initial_sdf == sys.argv[2]
    res_xyz = INSERT_HERE
    assert res_xyz == sys.argv[3]
    ic(start_xyz, initial_sdf, res_xyz)

    NUM_PROCS = INSERT_HERE

    assert os.path.isfile(start_xyz), f"Start XYZ named '{start_xyz}' not found"
    assert os.path.isfile(initial_sdf), f"Initial SDF named '{initial_sdf}' not found"

    # Prepare
    initial_p = Confpool()
    try:
        initial_p.include_from_file(start_xyz)
    except:
        print(f"OPT FAILED on '{start_xyz}'")
        ic(start_xyz, initial_sdf, res_xyz)
        sys.exit(0)
    ConformerInfo.standardize_confpool_descriptions(initial_p)
    
    if NUM_PROCS == 1 or len(initial_p) < 100:
        optimize_gfnff_and_geomcheck(
            initial_p=initial_p,
            initial_sdf=initial_sdf,
            res_xyz=res_xyz,
            xtb_dir=XTB_TEMP_DIR,
        )
    else:
        assert NUM_PROCS < len(initial_p)
        initial_p_list = confpool_to_batches(initial_p, NUM_PROCS)
        thread_data = []
        outputs_in_order = []
        for i, cur_p in enumerate(initial_p_list):
            input_name = os.path.join(XYZ_TEMP_DIR, f'input_{i}.xyz')
            output_name = os.path.join(XYZ_TEMP_DIR, f'output_{i}.xyz')
            cur_p.save_xyz(input_name)
            thread_data.append([
                input_name,
                initial_sdf,
                output_name
            ])
            outputs_in_order.append(output_name)
        
        multiproc_gfnff_opt(
            args_list=thread_data,
            num_proc=NUM_PROCS,
            xtb_dir=XTB_TEMP_DIR
        )

        output_p = Confpool()
        for output_name in outputs_in_order:
            output_p.include_from_file(output_name)
        output_p.save_xyz(res_xyz)
    shutil.rmtree(XYZ_TEMP_DIR)
    shutil.rmtree(XTB_TEMP_DIR)

def merge_optimized_ensembles(
    result_item,
    **input_items,
) -> None:
    for opttype, input_item in input_items.items():
        for xyz_name, keys in input_item:
            result_keys = {'opttype': opttype, **keys}
            result_path = result_item.get_path(**result_keys)
            shutil.copy2(xyz_name, result_path)
            result_item.include_element(result_path)

def filter_ensembles():
    import sys, os, ntpath, json
    module_dir = INSERT_HERE
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    os.chdir(os.path.dirname(__file__))

    import ringo
    from chemscripts import geom as ccgeom
    from utils.confsearch import ConformerInfo
    
    from icecream import install
    install()

    PYXYZ_SETTINGS = INSERT_HERE
    USE_ENERGY_CUTOFF = INSERT_HERE
    ENERGY_KEY = 'Energy'
    INDEX_KEY = 'Index'
    
    # Unpack the 'input_data' dict
    optimized_xyz = INSERT_HERE
    assert optimized_xyz == sys.argv[1]
    filtered_xyz = INSERT_HERE
    assert filtered_xyz == sys.argv[3]
    topology_sdf = INSERT_HERE
    assert topology_sdf == sys.argv[4]

    num_automorphisms = int(sys.argv[2])
    assert os.path.isfile(optimized_xyz), f"Start XYZ '{optimized_xyz}' not found"
    assert os.path.isfile(topology_sdf), f"Start SDF '{topology_sdf}' not found"

    filtered_p = ringo.Confpool()
    filtered_p.include_from_file(optimized_xyz)
    filtered_p.filter(lambda m: ConformerInfo(description=m.descr).is_successful)
    filtered_p['Index'] = lambda m: ConformerInfo(description=m.descr).index
    if USE_ENERGY_CUTOFF:
        filtered_p[ENERGY_KEY] = lambda m: ConformerInfo(description=m.descr).energy

    ignore_elements: list[str | int]
    if num_automorphisms < 1000: # and 'PRDCC002404' not in topology_sdf:
        ignore_elements = ['HCarbon']
    else:
        mol = ringo.Molecule(sdf=topology_sdf)
        biggest_frag_atoms: list[int] = mol.get_biggest_ringfrag_atoms()
        ignore_elements = [node for node in mol.molgraph_access().nodes if node not in biggest_frag_atoms]
    filtered_p.generate_connectivity(0, mult=1.3, ignore_elements=ignore_elements)
    filtered_p.generate_isomorphisms()
    
    filtering_kwargs = {key: value for key, value in PYXYZ_SETTINGS.items() if key in (
        'mirror_match',
        'print_status',
    )}
    if USE_ENERGY_CUTOFF:
        filtering_kwargs['energy_threshold'] = 1.0
        filtering_kwargs['energy_key'] = ENERGY_KEY
    initial_size: int = len(filtered_p)
    stats: dict[str, int] = filtered_p.rmsd_filter(PYXYZ_SETTINGS['rmsd_cutoff'], **filtering_kwargs)
    n_deleted: int = stats['DelCount']
    assert initial_size - len(filtered_p) == n_deleted
    unique_conformers_idxs = [int(idx) for idx in filtered_p[INDEX_KEY]]

    resulting_p = ringo.Confpool()
    resulting_p.include_from_file(optimized_xyz)
    resulting_p[INDEX_KEY] = lambda m: ConformerInfo(description=m.descr).index

    for m in resulting_p:
        if int(m[INDEX_KEY]) in unique_conformers_idxs:
            continue

        old_descr: str = m.descr
        conformer_info = ConformerInfo(description=old_descr)
        if conformer_info.is_successful:
            m.descr = ConformerInfo.updated_status(
                old_descr=old_descr,
                update='rmsdfail'
            )
    resulting_p.save_xyz(filtered_xyz)


def benchmark_transforms(ds, main_logger, execscripts_dir, maxproc=1) -> list[Transform]:
    # BENCHMARK_TIMES = [10, 60, 600]
    # BENCHMARK_TIMES = [10, 60, 600]
    BENCHMARK_TIMES = [10, 60]
    # BENCHMARK_TIMES = [600] # CHECK Full times

    assert_availability('exec_dummycrest.sh')
    assert_availability('exec_xtbmtd.sh')
    assert_availability('exec_xtbopt.sh')
    
    RINGO_METHOD = 'ringo'
    RINGO_VS_RDKIT_METHOD = 'ringo-vs-rdkit'
    RINGO_VS_RDKIT2024_METHOD = 'ringo-vs-rdkit2024'
    RINGO_VS_MTD_METHOD = 'ringo-vs-mtd'
    RINGO_VS_CREST_METHOD = 'ringo-vs-crest'
    RINGO_VS_MMRING_METHOD = 'ringo-vs-mmring'
    
    RINGO_CN_METHOD = 'ringoCN'
    RINGO_CN_VS_RDKIT_METHOD = 'ringoCN-vs-rdkit'
    RINGO_CN_VS_MTD_METHOD = 'ringoCN-vs-mtd'
    RINGO_CN_VS_CREST_METHOD = 'ringoCN-vs-crest'
    
    RDKIT_METHOD = 'ETKDGv3'
    RDKITv1_METHOD = 'ETKDGv1'
    RDKITv3_2024_METHOD = 'ETKDGv3-2024'
    MTD_METHOD = 'mtd'

    MMFF_LEVEL = 'MMFF-RDKit'
    GFNFF_LEVEL = 'GFNFF-XTB'
    
    PARTIAL_SAMPLING_DATA = [
        # Reference
        # {'xyz': 'rdkit_ensemble_xyz',           'stats': 'rdkit_stats_single'},
        {'xyz': 'rdkitv1_ensemble_xyz',         'stats': 'rdkitv1_stats_single'},
        {'xyz': 'rdkitv3_2024_ensemble_xyz',    'stats': 'rdkitv3_2024_stats_single'},
        {'xyz': 'mtd_ensemble_xyz',             'stats': 'mtd_stats_single'},
        
        # MCR without special amide bond treatment
        {'xyz': 'mcr_ensemble_xyz',             'stats': 'mcr_stats_single'},
        {'xyz': 'mcr_vs_rdkit_ensemble_xyz',    'stats': 'mcr_vs_rdkit_stats_single'},
        {'xyz': 'mcr_vs_mtd_ensemble_xyz',      'stats': 'mcr_vs_mtd_stats_single'},
        {'xyz': 'mcr_vs_crest_ensemble_xyz',    'stats': 'mcr_vs_crest_stats_single'},
        {'xyz': 'mcr_vs_rdkit2024_ensemble_xyz','stats': 'mcr_vs_rdkit2024_stats_single'},
        {'xyz': 'mcr_vs_mmring_ensemble_xyz',    'stats': 'mcr_vs_mmring_stats_single'},
      
        # MCR with E,Z-amide bond sampling
        # {'xyz': 'mcrCN_ensemble_xyz',           'stats': 'mcrCN_stats_single'},
        # {'xyz': 'mcrCN_vs_rdkit_ensemble_xyz',  'stats': 'mcrCN_vs_rdkit_stats_single'},
        # {'xyz': 'mcrCN_vs_mtd_ensemble_xyz',    'stats': 'mcrCN_vs_mtd_stats_single'},
        # {'xyz': 'mcrCN_vs_crest_ensemble_xyz',  'stats': 'mcrCN_vs_crest_stats_single'},

        # Ensembles obtained outside pipeline framework
        # {'xyz': 'old_ensemble_xyz',             'stats': 'old_stats_single'},
        {'xyz': 'heavy_old_ensemble_xyz',       'stats': 'heavy_old_stats_single'},
    ]

    METHODS_REQUIRE_REORDERING = [
        'rdkitOld',
        # RDKIT_METHOD,
        RDKITv3_2024_METHOD,
        RDKITv1_METHOD,
    ]

    ENSEMBLE_COMBINE_SOURCES = [
        value
        for element in PARTIAL_SAMPLING_DATA
        for value in element.values()
    ]


    def inject_old_entries(
            old_sampling_entries_fixed, old_sampling_xyz, old_sampling_cpuload_total,
            old_ensemble_xyz, old_stats_single,
            heavy_old_ensemble_xyz, heavy_old_stats_single
        ):
        timelimit_mapping = {
            keys['old_method']: keys['timelimit']
            for element, keys in old_sampling_entries_fixed
        }
        old_methods = set(timelimit_mapping.keys())

        for element, keys in old_sampling_entries_fixed:
            assert element['old_method'] == keys['old_method'], \
                f"{element['old_method']} vs. {keys['old_method']} in {keys}"

        def inject_old(old_method, new_method, timelimit):
            new_method += 'Old'

            if timelimit == 'long':
                target_ensemble_item = heavy_old_ensemble_xyz
                target_stats_item = heavy_old_stats_single
            else:
                target_ensemble_item = old_ensemble_xyz
                target_stats_item = old_stats_single

            old_path = old_sampling_xyz.access_element(old_method=old_method, timelimit=timelimit)
            new_path = target_ensemble_item.get_path(method=new_method, timelimit=timelimit)
            shutil.copy2(old_path, new_path)
            target_ensemble_item.include_element(new_path, method=new_method, timelimit=timelimit)

            old_stat_entry = old_sampling_entries_fixed.access_element(old_method=old_method, timelimit=timelimit)
            fixed_stat = {
                key if key != 'old_method' else 'method': value if key != 'old_method' else new_method
                for key, value in old_stat_entry.items()
            }
            cpuload: float = old_sampling_cpuload_total.access_element(old_method=old_method)
            if timelimit != 'long':
                assert cpuload == 1, f"Unexpected CPU load={cpuload} for {old_stat_entry}"
            fixed_stat['time'] *= cpuload

            stats_path = target_stats_item.get_path(method=new_method, timelimit=timelimit)
            with open(stats_path, 'w') as f:
                json.dump(fixed_stat, f)    
            target_stats_item.include_element(stats_path, method=new_method, timelimit=timelimit)
        
        if 'crestfailed' in old_methods:
            inject_old(old_method='crestfailed', new_method='crest', timelimit=timelimit_mapping['crestfailed'])
        elif 'crest' in old_methods: # Load 'crest' only when there is no 'crestfailed'
            inject_old(old_method='crest', new_method='crest', timelimit=timelimit_mapping['crest'])
        
        assert 'ringointel' in old_methods
        inject_old(old_method='ringointel', new_method='ringo', timelimit=timelimit_mapping['ringointel'])

        for old_method in old_methods:
            # These do not require any special naming logic
            if old_method not in ('crest', 'crestfailed', 'ringointel'):
                inject_old(old_method=old_method, new_method=old_method, timelimit=timelimit_mapping[old_method])


    def get_universal_subs(ensemble_item, stats_item, specific_subs: dict, custom_termination: Optional[dict[str, any]]={}, **kw):
        """Generate substitutions for arbitrary method's sampling script (e.g. 'ringo_benchmark' function)

        Args:
            ensemble_item (FileItem): ensemble files to be created
            stats_item (FileItem): sampling run stats to be created

        Returns:
            Dict[str, Any]: substitutions for the INSERT_HERE parts of the script
        """

        long_timelimit = 'long' in custom_termination
        if long_timelimit:
            assert len(custom_termination) == 1, \
                f"No other modes are allowed in case of timelimit comparison with method that has timelimit='long'"

        # These blocks 'benchmark_times' and 'sampling_modes' account for logic branching
        # when I may want to have sampling_modes[int] and sample for 'int' seconds or
        # when doing comparison with CREST/MM I may want to have sampling_modes['long'] and sample for custom_termination['long'] seconds
        if long_timelimit:
            benchmark_times = ['long']
        else:
            benchmark_times = BENCHMARK_TIMES

        sampling_modes = {
            timelimit: {
                'ensemble': ensemble_item.get_path(timelimit=timelimit),
                'stats': stats_item.get_path(timelimit=timelimit),
                'termination':
                    {'timelimit': timelimit} if len(custom_termination) == 0
                    else custom_termination[timelimit]
            }
            for timelimit in benchmark_times
        }
        return { 'MODES': sampling_modes, **specific_subs }
    

    def universal_load_results(ensemble_item, stats_item, long_timelimit=False, **kw): # long??????????
        """Load ensemble xyz and stats into specified items

        Args:
            ensemble_item (FileItem): ensemble files that were created
            stats_item (FileItem): sampling run stats that were created
        """
        if long_timelimit:
            benchmark_times = ['long']
        else:
            benchmark_times = BENCHMARK_TIMES
        
        for timelimit in benchmark_times:
            stats_name = stats_item.get_path(timelimit=timelimit)
            if os.path.exists(stats_name):
                stats_item.include_element(stats_name)

            xyz_name = ensemble_item.get_path(timelimit=timelimit)
            if os.path.exists(xyz_name):
                ensemble_item.include_element(xyz_name)


    def lazy_sampling(stats_item_name: str, timelimit_key: str) -> Callable[..., bool]:
        def skip_function(items) -> bool:
            """Skips sampling run if stat files are already present for required timelimits

            Args:
                items (Dict[str, DataItem]): dict of all DataItem restrictions before transform call itself.

            Returns:
                bool: True - skip, False - execute.
            """
            required_times = set(BENCHMARK_TIMES)
            done_times = set(keys[timelimit_key] for file, keys in items[stats_item_name])
            return required_times.issubset(done_times) or 'long' in done_times
        return skip_function


    def combine_sampling_data(**kw):
        for cur_items in PARTIAL_SAMPLING_DATA:
            xyz_item = kw[cur_items['xyz']]
            stats_item = kw[cur_items['stats']]

            for stats_file, stats_keys in stats_item:
                if stats_keys['timelimit'] != 'long' and stats_keys['timelimit'] not in BENCHMARK_TIMES:
                    continue

                stats_keys_stripped = {
                    key: value
                    for key, value in stats_keys.items()
                    if key != 'method'
                }

                with open(stats_file, 'r') as f:
                    method_name = json.load(f)['method']
                
                if xyz_item.element_exists(**stats_keys):
                    raw_ensemble_file = kw['raw_ensemble_xyz'].get_path(method=method_name, **stats_keys_stripped)
                    shutil.copy2(xyz_item.access_element(**stats_keys), raw_ensemble_file)
                    kw['raw_ensemble_xyz'].include_element(raw_ensemble_file, method=method_name, **stats_keys_stripped)

                new_stats_file = kw['csrun_stats_single'].get_path(method=method_name, **stats_keys_stripped)
                shutil.copy2(stats_file, new_stats_file)
                kw['csrun_stats_single'].include_element(new_stats_file, method=method_name, **stats_keys_stripped)


    def verify_testmols(optimized_testmols_sdfs, randomized_testmols_sdfs, testcase) -> bool:
        from ringo import Confpool
        from utils.confsearch import TopologyChecker
        
        p = Confpool()
        
        start_mol = ccgeom.Molecule(sdf=optimized_testmols_sdfs)
        start_xyz, start_sym = start_mol.as_xyz()
        p.include_from_xyz(start_xyz, '')
        p.atom_symbols = start_sym
        
        random_mol = ccgeom.Molecule(sdf=randomized_testmols_sdfs)
        random_xyz, random_sym = random_mol.as_xyz()
        assert random_sym == start_sym
        p.include_from_xyz(random_xyz, '')
        p.atom_symbols = random_sym

        topo_checker = TopologyChecker(start_mol.G)
        p.generate_connectivity(0, mult=1.3)
        same_topology, extra_bonds, missing_bonds = topo_checker.same_topology(p.get_connectivity())
        assert same_topology, \
            f"testcase={testcase}, extra_bonds={extra_bonds}, missing_bonds={missing_bonds}"
        
        p.generate_connectivity(1, mult=1.3)
        same_topology, extra_bonds, missing_bonds = topo_checker.same_topology(p.get_connectivity())
        return same_topology


    benchmark_run_transforms: list[Transform] = [
        # # Just a late verification that testcase's topologies are okay
        # templates.map('verify_correct_topology',
        #     input=['optimized_testmols_sdfs', 'randomized_testmols_sdfs'], output='random_testmols_correct',
        #     mapping=verify_testmols, aware_keys=['testcase']
        # ),

        # # Load ensembles and stats that were generated outside of pipeline framework
        # templates.pd_from_csv('load_old_sampling_stats',
        #     input='old_sampling_csv', output='old_sampling_df'
        # ),
        # templates.extract_df_rows('extract_old_sampling_rows',
        #     input='old_sampling_df', output='old_sampling_entries',
        #     column_to_key={'method': 'old_method'},
        #     value_mapping={'testcase': lambda old_value: old_value.replace('_', '')},
        #     store_all_keys_in_elements=True
        # ),
        # templates.map('add_timelimit_to_old_sampling_rows',
        #     input='old_sampling_entries', output='old_sampling_entries_fixed',
        #     mapping=lambda old_sampling_entries, **kw: (
        #         x for x in ((old_sampling_entries, {
        #             'timelimit': 600 if old_sampling_entries['old_method'] in ('ringointel', 'rdkit', 'mtd')
        #                 else 'long'
        #         }),)
        #     )
        # ),
        # templates.load_json('old_load_avg_cpuload',
        #     input='old_sampling_cpuload_json', output='old_sampling_cpuload_avg',
        #     post_mapping=lambda object, keys: (sum(object) / len(object), {'testcase': keys['testcase'].replace('_', '')})
        # ),
        # templates.map('generate_old_singlethread_cpuload',
        #     input='old_sampling_entries', output='old_sampling_singlethread_cpuload',
        #     mapping=lambda **kw: 1.0
        # ),
        # templates.substitute('old_finalize_cpuload',
        #     input='old_sampling_cpuload_avg',
        #     substituent='old_sampling_singlethread_cpuload',
        #     output='old_sampling_cpuload_total',
        #     merged_keys=['testset', 'testcase', 'old_method']
        # ),
        # templates.exec('inject_old_entries_into_generated',
        #     input=['old_sampling_entries_fixed', 'old_sampling_xyz', 'old_sampling_cpuload_total'],
        #     output=['old_ensemble_xyz', 'old_stats_single', 'heavy_old_ensemble_xyz', 'heavy_old_stats_single'],
        #     merged_keys=['old_method', 'timelimit'],
        #     method=inject_old_entries
        # ),

        # templates.restrict('extract_heavy_testmols_sdfs',
        #     input='randomized_testmols_sdfs', ref='heavy_old_stats_single', output='randomized_heavy_testmols_sdfs',
        #     merged_keys=['method', 'testcase', 'timelimit']
        # ),
        templates.restrict('extract_heavy_final_summary_object',
            input='final_summary_object', ref='heavy_old_stats_single', output='heavy_final_summary_object',
            merged_keys=['method', 'testcase', 'timelimit']
        ),

        # # Generate & process ensembles
        templates.pyfunction_subprocess('mcr_benchmark',
            input=['randomized_testmols_sdfs', 'final_summary_object'], output=['mcr_ensemble_xyz', 'mcr_stats_single'],
            pyfunction=ringo_benchmark, calcdir='calcdir', nproc=int(maxproc/4),
            argv_prepare=lambda randomized_testmols_sdfs, final_summary_object, **kw:
                (randomized_testmols_sdfs.access_element(), final_summary_object.access_element()['num_automorphisms']),
            subs=lambda mcr_ensemble_xyz, mcr_stats_single, **kw: get_universal_subs(
                ensemble_item=mcr_ensemble_xyz,
                stats_item=mcr_stats_single,
                specific_subs={'RINGO_METHOD': RINGO_METHOD, 'CONSTRAIN_AMIDE_BONDS': False, 'CUSTOM_TERMINATION': None},
                **kw
            ),
            output_process=lambda mcr_ensemble_xyz, mcr_stats_single, **kw: universal_load_results(
                ensemble_item=mcr_ensemble_xyz,
                stats_item=mcr_stats_single,
                **kw
            )
        ), #.greedy_on(lazy_sampling(stats_item_name='mcr_stats_single', timelimit_key='timelimit')),

        # templates.pyfunction_subprocess('mcrCN_benchmark',
        #     input=['randomized_testmols_sdfs', 'final_summary_object'], output=['mcrCN_ensemble_xyz', 'mcrCN_stats_single'],
        #     pyfunction=ringo_benchmark, calcdir='calcdir', nproc=maxproc,
        #     argv_prepare=lambda randomized_testmols_sdfs, final_summary_object, **kw:
        #         (randomized_testmols_sdfs.access_element(), final_summary_object.access_element()['num_automorphisms']),
        #     subs=lambda mcrCN_ensemble_xyz, mcrCN_stats_single, **kw: get_universal_subs(
        #         ensemble_item=mcrCN_ensemble_xyz,
        #         stats_item=mcrCN_stats_single,
        #         specific_subs={'RINGO_METHOD': RINGO_CN_METHOD, 'CONSTRAIN_AMIDE_BONDS': True, 'CUSTOM_TERMINATION': None},
        #         **kw
        #     ),
        #     output_process=lambda mcrCN_ensemble_xyz, mcrCN_stats_single, **kw: universal_load_results(
        #         ensemble_item=mcrCN_ensemble_xyz,
        #         stats_item=mcrCN_stats_single,
        #         **kw
        #     )
        # ).greedy_on(lazy_sampling(stats_item_name='mcrCN_stats_single', timelimit_key='timelimit')),
        
        # templates.pyfunction_subprocess('rdkit_benchmark',
        #     input=['randomized_testmols_sdfs', 'final_summary_object'], output=['rdkit_ensemble_xyz', 'rdkit_stats_single'],
        #     pyfunction=rdkit_benchmark, calcdir='calcdir', nproc=maxproc,
        #     argv_prepare=lambda randomized_testmols_sdfs, **kw:
        #         (randomized_testmols_sdfs.access_element(),),
        #     subs=lambda final_summary_object, rdkit_ensemble_xyz, rdkit_stats_single, **kw: get_universal_subs(
        #         ensemble_item=rdkit_ensemble_xyz,
        #         stats_item=rdkit_stats_single,
        #         specific_subs={'RDKIT_METHOD': RDKIT_METHOD, 'SMILES': final_summary_object.access_element()['smiles']},
        #         **kw
        #     ),
        #     output_process=lambda rdkit_ensemble_xyz, rdkit_stats_single, **kw: universal_load_results(
        #         ensemble_item=rdkit_ensemble_xyz,
        #         stats_item=rdkit_stats_single,
        #         **kw
        #     )
        # ).greedy_on(lazy_sampling(stats_item_name='rdkit_stats_single', timelimit_key='timelimit')),

        # templates.pyfunction_subprocess('rdkit_2022_benchmark',
        #     input=['randomized_testmols_sdfs', 'final_summary_object'], output=['rdkitv1_ensemble_xyz', 'rdkitv1_stats_single'],
        #     pyfunction=rdkit_benchmark, calcdir='calcdir', nproc=maxproc/2,
        #     argv_prepare=lambda randomized_testmols_sdfs, **kw:
        #         (randomized_testmols_sdfs.access_element(),),
        #     subs=lambda final_summary_object, rdkitv1_ensemble_xyz, rdkitv1_stats_single, **kw: get_universal_subs(
        #         ensemble_item=rdkitv1_ensemble_xyz,
        #         stats_item=rdkitv1_stats_single,
        #         specific_subs={'RDKIT_METHOD': RDKITv1_METHOD, 'SMILES': final_summary_object.access_element()['smiles']},
        #         **kw
        #     ),
        #     output_process=lambda rdkitv1_ensemble_xyz, rdkitv1_stats_single, **kw: universal_load_results(
        #         ensemble_item=rdkitv1_ensemble_xyz,
        #         stats_item=rdkitv1_stats_single,
        #         **kw
        #     ),
        #     env_activation=RDKIT2022_ACTIVATION
        # ).greedy_on(lazy_sampling(stats_item_name='rdkitv1_stats_single', timelimit_key='timelimit')),

        templates.pyfunction_subprocess('rdkit_2024_benchmark',
            input=['randomized_testmols_sdfs', 'final_summary_object'], output=['rdkitv3_2024_ensemble_xyz', 'rdkitv3_2024_stats_single'],
            pyfunction=rdkit_benchmark, calcdir='calcdir', nproc=int(maxproc/2),
            argv_prepare=lambda randomized_testmols_sdfs, **kw:
                (randomized_testmols_sdfs.access_element(),),
            subs=lambda final_summary_object, rdkitv3_2024_ensemble_xyz, rdkitv3_2024_stats_single, **kw: get_universal_subs(
                ensemble_item=rdkitv3_2024_ensemble_xyz,
                stats_item=rdkitv3_2024_stats_single,
                specific_subs={'RDKIT_METHOD': RDKITv3_2024_METHOD, 'SMILES': final_summary_object.access_element()['smiles']},
                **kw
            ),
            output_process=lambda rdkitv3_2024_ensemble_xyz, rdkitv3_2024_stats_single, **kw: universal_load_results(
                ensemble_item=rdkitv3_2024_ensemble_xyz,
                stats_item=rdkitv3_2024_stats_single,
                **kw
            ),
            env_activation=RDKIT2024_ACTIVATION
        ).greedy_on(lazy_sampling(stats_item_name='rdkitv3_2024_stats_single', timelimit_key='timelimit')),

        templates.pyfunction_subprocess('mtd_benchmark',
            input=['randomized_testmols_sdfs', 'final_summary_object'], output=['mtd_ensemble_xyz', 'mtd_stats_single'],
            pyfunction=mtd_benchmark, calcdir='calcdir', nproc=maxproc,
            argv_prepare=lambda randomized_testmols_sdfs, **kw:
                (randomized_testmols_sdfs.access_element(),),
            subs=lambda final_summary_object, mtd_ensemble_xyz, mtd_stats_single, **kw: get_universal_subs(
                ensemble_item=mtd_ensemble_xyz,
                stats_item=mtd_stats_single,
                specific_subs={'MTD_METHOD': MTD_METHOD, 'EXECSCRIPTS_DIR': execscripts_dir},
                **kw
            ),
            output_process=lambda mtd_ensemble_xyz, mtd_stats_single, **kw: universal_load_results(
                ensemble_item=mtd_ensemble_xyz,
                stats_item=mtd_stats_single,
                **kw
            )
        ).greedy_on(lazy_sampling(stats_item_name='mtd_stats_single', timelimit_key='timelimit')),

        # # Weird termination conditions to compare against RDKit & MTD
        # templates.pyfunction_subprocess('mcr_vs_rdkit_benchmark',
        #     input=['randomized_testmols_sdfs', 'final_summary_object', 'rdkit_stats_single'],
        #     output=['mcr_vs_rdkit_ensemble_xyz', 'mcr_vs_rdkit_stats_single'],
        #     pyfunction=ringo_benchmark, calcdir='calcdir', nproc=maxproc, merged_keys=['timelimit'],
        #     argv_prepare=lambda randomized_testmols_sdfs, final_summary_object, **kw:
        #         (randomized_testmols_sdfs.access_element(), final_summary_object.access_element()['num_automorphisms']),
        #     subs=lambda mcr_vs_rdkit_ensemble_xyz, mcr_vs_rdkit_stats_single, rdkit_stats_single, **kw: get_universal_subs(
        #         ensemble_item=mcr_vs_rdkit_ensemble_xyz,
        #         stats_item=mcr_vs_rdkit_stats_single,
        #         specific_subs={
        #             'RINGO_METHOD': RINGO_VS_RDKIT_METHOD,
        #             'CONSTRAIN_AMIDE_BONDS': False,
        #         },
        #         custom_termination={
        #             timelimit: {
        #                 'max_conformers': confsearch.json_from_file(rdkit_stats_single.access_element(timelimit=timelimit))['nunique'],
        #                 'timelimit': 3600,
        #             }
        #             for timelimit in BENCHMARK_TIMES
        #         },
        #         **kw
        #     ),
        #     output_process=lambda mcr_vs_rdkit_ensemble_xyz, mcr_vs_rdkit_stats_single, **kw: universal_load_results(
        #         ensemble_item=mcr_vs_rdkit_ensemble_xyz,
        #         stats_item=mcr_vs_rdkit_stats_single,
        #         **kw
        #     )
        # ).greedy_on(lazy_sampling(stats_item_name='mcr_vs_rdkit_stats_single', timelimit_key='timelimit')),
        templates.pyfunction_subprocess('mcr_vs_rdkit2024_benchmark',
            input=['randomized_testmols_sdfs', 'final_summary_object', 'rdkitv3_2024_stats_single'],
            output=['mcr_vs_rdkit2024_ensemble_xyz', 'mcr_vs_rdkit2024_stats_single'],
            pyfunction=ringo_benchmark, calcdir='calcdir', nproc=int(maxproc/4), merged_keys=['timelimit'],
            argv_prepare=lambda randomized_testmols_sdfs, final_summary_object, **kw:
                (randomized_testmols_sdfs.access_element(), final_summary_object.access_element()['num_automorphisms']),
            subs=lambda mcr_vs_rdkit2024_ensemble_xyz, mcr_vs_rdkit2024_stats_single, rdkitv3_2024_stats_single, **kw: get_universal_subs(
                ensemble_item=mcr_vs_rdkit2024_ensemble_xyz,
                stats_item=mcr_vs_rdkit2024_stats_single,
                specific_subs={
                    'RINGO_METHOD': RINGO_VS_RDKIT2024_METHOD,
                    'CONSTRAIN_AMIDE_BONDS': False,
                },
                custom_termination={
                    timelimit: {
                        'max_conformers': confsearch.json_from_file(rdkitv3_2024_stats_single.access_element(timelimit=timelimit))['nunique'],
                        'timelimit': 3600,
                    }
                    for timelimit in BENCHMARK_TIMES
                },
                **kw
            ),
            output_process=lambda mcr_vs_rdkit2024_ensemble_xyz, mcr_vs_rdkit2024_stats_single, **kw: universal_load_results(
                ensemble_item=mcr_vs_rdkit2024_ensemble_xyz,
                stats_item=mcr_vs_rdkit2024_stats_single,
                **kw
            )
        ).greedy_on(lazy_sampling(stats_item_name='mcr_vs_rdkit2024_stats_single', timelimit_key='timelimit')),
        
        # templates.pyfunction_subprocess('mcr_vs_mtd_benchmark',
        #     input=['randomized_testmols_sdfs', 'final_summary_object', 'mtd_stats_single'],
        #     output=['mcr_vs_mtd_ensemble_xyz', 'mcr_vs_mtd_stats_single'],
        #     pyfunction=ringo_benchmark, calcdir='calcdir', nproc=maxproc, merged_keys=['timelimit'],
        #     argv_prepare=lambda randomized_testmols_sdfs, final_summary_object, **kw:
        #         (randomized_testmols_sdfs.access_element(), final_summary_object.access_element()['num_automorphisms']),
        #     subs=lambda mcr_vs_mtd_ensemble_xyz, mcr_vs_mtd_stats_single, mtd_stats_single, **kw: get_universal_subs(
        #         ensemble_item=mcr_vs_mtd_ensemble_xyz,
        #         stats_item=mcr_vs_mtd_stats_single,
        #         specific_subs={
        #             'RINGO_METHOD': RINGO_VS_MTD_METHOD,
        #             'CONSTRAIN_AMIDE_BONDS': False,
        #         },
        #         custom_termination={
        #             timelimit: {
        #                 'max_conformers': confsearch.json_from_file(mtd_stats_single.access_element(timelimit=timelimit))['nunique'],
        #                 'timelimit': 3600,
        #             }
        #             for timelimit in BENCHMARK_TIMES
        #         },
        #         **kw
        #     ),
        #     output_process=lambda mcr_vs_mtd_ensemble_xyz, mcr_vs_mtd_stats_single, **kw: universal_load_results(
        #         ensemble_item=mcr_vs_mtd_ensemble_xyz,
        #         stats_item=mcr_vs_mtd_stats_single,
        #         **kw
        #     )
        # ).greedy_on(lazy_sampling(stats_item_name='mcr_vs_mtd_stats_single', timelimit_key='timelimit')),
        
        # templates.pyfunction_subprocess('mcr_vs_crest_benchmark',
        #     input=['randomized_heavy_testmols_sdfs', 'heavy_final_summary_object', 'heavy_old_stats_single'],
        #     output=['mcr_vs_crest_ensemble_xyz', 'mcr_vs_crest_stats_single'],
        #     pyfunction=ringo_benchmark, calcdir='calcdir', nproc=maxproc, merged_keys=['timelimit', 'method'],
        #     argv_prepare=lambda randomized_heavy_testmols_sdfs, heavy_final_summary_object, **kw:
        #         (randomized_heavy_testmols_sdfs.access_element(), heavy_final_summary_object.access_element()['num_automorphisms']),
        #     subs=lambda mcr_vs_crest_ensemble_xyz, mcr_vs_crest_stats_single, heavy_old_stats_single, **kw: get_universal_subs(
        #         ensemble_item=mcr_vs_crest_ensemble_xyz,
        #         stats_item=mcr_vs_crest_stats_single,
        #         specific_subs={
        #             'RINGO_METHOD': RINGO_VS_CREST_METHOD,
        #             'CONSTRAIN_AMIDE_BONDS': False,
        #         },
        #         custom_termination={
        #             'long': {
        #                 'timelimit': confsearch.json_from_file(heavy_old_stats_single.access_element(method='crestOld', timelimit='long'))['time']
        #             }
        #         },
        #         **kw
        #     ),
        #     output_process=lambda mcr_vs_crest_ensemble_xyz, mcr_vs_crest_stats_single, **kw: universal_load_results(
        #         ensemble_item=mcr_vs_crest_ensemble_xyz,
        #         stats_item=mcr_vs_crest_stats_single,
        #         long_timelimit=True, # So it knows to check for timelimit='long' only!
        #         **kw
        #     )
        # ).greedy_on(lazy_sampling(stats_item_name='mcr_vs_crest_stats_single', timelimit_key='timelimit')),
        templates.pyfunction_subprocess('mcr_vs_mmring_benchmark',
            input=['randomized_heavy_testmols_sdfs', 'heavy_final_summary_object', 'heavy_old_stats_single'],
            output=['mcr_vs_mmring_ensemble_xyz', 'mcr_vs_mmring_stats_single'],
            pyfunction=ringo_benchmark, calcdir='calcdir', nproc=int(maxproc/4), merged_keys=['timelimit', 'method'],
            argv_prepare=lambda randomized_heavy_testmols_sdfs, heavy_final_summary_object, **kw:
                (randomized_heavy_testmols_sdfs.access_element(), heavy_final_summary_object.access_element()['num_automorphisms']),
            subs=lambda mcr_vs_mmring_ensemble_xyz, mcr_vs_mmring_stats_single, heavy_old_stats_single, **kw: get_universal_subs(
                ensemble_item=mcr_vs_mmring_ensemble_xyz,
                stats_item=mcr_vs_mmring_stats_single,
                specific_subs={
                    'RINGO_METHOD': RINGO_VS_MMRING_METHOD,
                    'CONSTRAIN_AMIDE_BONDS': False,
                },
                custom_termination={
                    'long': {
                        'timelimit': confsearch.json_from_file(heavy_old_stats_single.access_element(method='mmringOld', timelimit='long'))['time']
                    }
                },
                **kw
            ),
            output_process=lambda mcr_vs_mmring_ensemble_xyz, mcr_vs_mmring_stats_single, **kw: universal_load_results(
                ensemble_item=mcr_vs_mmring_ensemble_xyz,
                stats_item=mcr_vs_mmring_stats_single,
                long_timelimit=True, # So it knows to check for timelimit='long' only!
                **kw
            )
        ).greedy_on(lazy_sampling(stats_item_name='mcr_vs_mmring_stats_single', timelimit_key='timelimit')),

        # # The same as above just CONSTRAIN_AMIDE_BONDS is enabled
        # templates.pyfunction_subprocess('mcrCN_vs_rdkit_benchmark',
        #     input=['randomized_testmols_sdfs', 'final_summary_object', 'rdkit_stats_single'],
        #     output=['mcrCN_vs_rdkit_ensemble_xyz', 'mcrCN_vs_rdkit_stats_single'],
        #     pyfunction=ringo_benchmark, calcdir='calcdir', nproc=maxproc, merged_keys=['timelimit'],
        #     argv_prepare=lambda randomized_testmols_sdfs, final_summary_object, **kw:
        #         (randomized_testmols_sdfs.access_element(), final_summary_object.access_element()['num_automorphisms']),
        #     subs=lambda mcrCN_vs_rdkit_ensemble_xyz, mcrCN_vs_rdkit_stats_single, rdkit_stats_single, **kw: get_universal_subs(
        #         ensemble_item=mcrCN_vs_rdkit_ensemble_xyz,
        #         stats_item=mcrCN_vs_rdkit_stats_single,
        #         specific_subs={
        #             'RINGO_METHOD': RINGO_CN_VS_RDKIT_METHOD,
        #             'CONSTRAIN_AMIDE_BONDS': True,
        #         },
        #         custom_termination={
        #             timelimit: {
        #                 'max_conformers': confsearch.json_from_file(rdkit_stats_single.access_element(timelimit=timelimit))['nunique'],
        #                 'timelimit': 3600,
        #             }
        #             for timelimit in BENCHMARK_TIMES
        #         },
        #         **kw
        #     ),
        #     output_process=lambda mcrCN_vs_rdkit_ensemble_xyz, mcrCN_vs_rdkit_stats_single, **kw: universal_load_results(
        #         ensemble_item=mcrCN_vs_rdkit_ensemble_xyz,
        #         stats_item=mcrCN_vs_rdkit_stats_single,
        #         **kw
        #     )
        # ).greedy_on(lazy_sampling(stats_item_name='mcrCN_vs_rdkit_stats_single', timelimit_key='timelimit')),
        
        # templates.pyfunction_subprocess('mcrCN_vs_mtd_benchmark',
        #     input=['randomized_testmols_sdfs', 'final_summary_object', 'mtd_stats_single'],
        #     output=['mcrCN_vs_mtd_ensemble_xyz', 'mcrCN_vs_mtd_stats_single'],
        #     pyfunction=ringo_benchmark, calcdir='calcdir', nproc=maxproc, merged_keys=['timelimit'],
        #     argv_prepare=lambda randomized_testmols_sdfs, final_summary_object, **kw:
        #         (randomized_testmols_sdfs.access_element(), final_summary_object.access_element()['num_automorphisms']),
        #     subs=lambda mcrCN_vs_mtd_ensemble_xyz, mcrCN_vs_mtd_stats_single, mtd_stats_single, **kw: get_universal_subs(
        #         ensemble_item=mcrCN_vs_mtd_ensemble_xyz,
        #         stats_item=mcrCN_vs_mtd_stats_single,
        #         specific_subs={
        #             'RINGO_METHOD': RINGO_CN_VS_MTD_METHOD,
        #             'CONSTRAIN_AMIDE_BONDS': True,
        #         },
        #         custom_termination={
        #             timelimit: {
        #                 'max_conformers': confsearch.json_from_file(mtd_stats_single.access_element(timelimit=timelimit))['nunique'],
        #                 'timelimit': 3600,
        #             }
        #             for timelimit in BENCHMARK_TIMES
        #         },
        #         **kw
        #     ),
        #     output_process=lambda mcrCN_vs_mtd_ensemble_xyz, mcrCN_vs_mtd_stats_single, **kw: universal_load_results(
        #         ensemble_item=mcrCN_vs_mtd_ensemble_xyz,
        #         stats_item=mcrCN_vs_mtd_stats_single,
        #         **kw
        #     )
        # ).greedy_on(lazy_sampling(stats_item_name='mcrCN_vs_mtd_stats_single', timelimit_key='timelimit')),
        
        # templates.pyfunction_subprocess('mcrCN_vs_crest_benchmark',
        #     input=['randomized_heavy_testmols_sdfs', 'heavy_final_summary_object', 'heavy_old_stats_single'],
        #     output=['mcrCN_vs_crest_ensemble_xyz', 'mcrCN_vs_crest_stats_single'],
        #     pyfunction=ringo_benchmark, calcdir='calcdir', nproc=maxproc, merged_keys=['timelimit', 'method'],
        #     argv_prepare=lambda randomized_heavy_testmols_sdfs, heavy_final_summary_object, **kw:
        #         (randomized_heavy_testmols_sdfs.access_element(), heavy_final_summary_object.access_element()['num_automorphisms']),
        #     subs=lambda mcrCN_vs_crest_ensemble_xyz, mcrCN_vs_crest_stats_single, heavy_old_stats_single, **kw: get_universal_subs(
        #         ensemble_item=mcrCN_vs_crest_ensemble_xyz,
        #         stats_item=mcrCN_vs_crest_stats_single,
        #         specific_subs={
        #             'RINGO_METHOD': RINGO_CN_VS_CREST_METHOD,
        #             'CONSTRAIN_AMIDE_BONDS': True,
        #         },
        #         custom_termination={
        #             'long': {
        #                 'timelimit': confsearch.json_from_file(heavy_old_stats_single.access_element(method='crestOld', timelimit='long'))['time']
        #             }
        #         },
        #         **kw
        #     ),
        #     output_process=lambda mcrCN_vs_crest_ensemble_xyz, mcrCN_vs_crest_stats_single, **kw: universal_load_results(
        #         ensemble_item=mcrCN_vs_crest_ensemble_xyz,
        #         stats_item=mcrCN_vs_crest_stats_single,
        #         long_timelimit=True, # So it knows to check for timelimit='long' only!
        #         **kw
        #     )
        # ).greedy_on(lazy_sampling(stats_item_name='mcrCN_vs_crest_stats_single', timelimit_key='timelimit')),

        # Merge all ensebles from various sources under one DataItem
        templates.exec('sampling_data_combiner',
            input=ENSEMBLE_COMBINE_SOURCES, output=['raw_ensemble_xyz','csrun_stats_single'],
            method=combine_sampling_data,
            merged_keys=['testset', 'testcase', 'method', 'timelimit']
        ),
        templates.restrict('get_nonzero_testmols',
            input='randomized_testmols_sdfs', ref='raw_ensemble_xyz', output='nonzero_testmols_sdf',
            merged_keys=['method', 'timelimit', 'testset', 'testcase'],
            ensure_one_to_one=False
        ),

        templates.pyfunction_subprocess('fix_rdkit_atom_ordering',
            input=['raw_ensemble_xyz', 'nonzero_testmols_sdf'], output='fixed_ordering_ensemble_xyz', aware_keys=['method', 'timelimit'],
            pyfunction=fix_atom_ordering, calcdir='calcdir', nproc=1,
            argv_prepare=lambda raw_ensemble_xyz, nonzero_testmols_sdf, fixed_ordering_ensemble_xyz, **kw:
                (raw_ensemble_xyz.access_element(), nonzero_testmols_sdf.access_element(), fixed_ordering_ensemble_xyz.get_path()),
            subs=lambda raw_ensemble_xyz, nonzero_testmols_sdf, fixed_ordering_ensemble_xyz, method, **kw: {
                'raw_xyz_path': raw_ensemble_xyz.access_element(),
                'initial_sdf_path': nonzero_testmols_sdf.access_element(),
                'fixed_xyz_path': fixed_ordering_ensemble_xyz.get_path(),
                'method': method,
                'METHODS_REQUIRE_REORDERING': METHODS_REQUIRE_REORDERING,
            },
            output_process=lambda fixed_ordering_ensemble_xyz, **kw: confsearch.load_if_exists(fixed_ordering_ensemble_xyz)
        ).greedy_on('fixed_ordering_ensemble_xyz'),
        templates.restrict('get_orderfix_nonzero_testmols',
            input='randomized_testmols_sdfs', ref='fixed_ordering_ensemble_xyz', output='orderfix_nonzero_testmols_sdf',
            merged_keys=['method', 'timelimit', 'testset', 'testcase'],
            ensure_one_to_one=False
        ),

        templates.pyfunction_subprocess('mmff_optimize_ensembles',
            input=['fixed_ordering_ensemble_xyz', 'orderfix_nonzero_testmols_sdf'], output='mmff_opt_ensemble_xyz', aware_keys=['method', 'timelimit'],
            pyfunction=mmff_optimize_ensemble, calcdir='calcdir', nproc=int(maxproc/4),
            argv_prepare=lambda fixed_ordering_ensemble_xyz, orderfix_nonzero_testmols_sdf, mmff_opt_ensemble_xyz, **kw:
                (fixed_ordering_ensemble_xyz.access_element(), orderfix_nonzero_testmols_sdf.access_element(), mmff_opt_ensemble_xyz.get_path()),
            subs=lambda fixed_ordering_ensemble_xyz, orderfix_nonzero_testmols_sdf, mmff_opt_ensemble_xyz, **kw: {
                'NUM_PROCS': int(maxproc/4),
                'start_xyz': fixed_ordering_ensemble_xyz.access_element(),
                'initial_sdf': orderfix_nonzero_testmols_sdf.access_element(),
                'res_xyz': mmff_opt_ensemble_xyz.get_path(),
            },
            output_process=lambda mmff_opt_ensemble_xyz, **kw: confsearch.load_if_exists(mmff_opt_ensemble_xyz)
        ).greedy_on('mmff_opt_ensemble_xyz'),
        templates.pyfunction_subprocess('gfnff_optimize_ensembles',
            input=['fixed_ordering_ensemble_xyz', 'orderfix_nonzero_testmols_sdf'], output='gfnff_opt_ensemble_xyz', aware_keys=['method', 'timelimit'],
            pyfunction=gfnff_optimize_ensemble, calcdir='calcdir', nproc=maxproc,
            argv_prepare=lambda fixed_ordering_ensemble_xyz, orderfix_nonzero_testmols_sdf, gfnff_opt_ensemble_xyz, **kw:
                (fixed_ordering_ensemble_xyz.access_element(), orderfix_nonzero_testmols_sdf.access_element(), gfnff_opt_ensemble_xyz.get_path()),
            subs=lambda fixed_ordering_ensemble_xyz, orderfix_nonzero_testmols_sdf, gfnff_opt_ensemble_xyz, **kw: {
                'EXECSCRIPTS_DIR': execscripts_dir,
                'NUM_PROCS': maxproc,
                'start_xyz': fixed_ordering_ensemble_xyz.access_element(),
                'initial_sdf': orderfix_nonzero_testmols_sdf.access_element(),
                'res_xyz': gfnff_opt_ensemble_xyz.get_path(),
            },
            output_process=lambda gfnff_opt_ensemble_xyz, **kw: confsearch.load_if_exists(gfnff_opt_ensemble_xyz)
        ).greedy_on('gfnff_opt_ensemble_xyz'),
        templates.pyfunction_subprocess('gfnff_postoptimize_ensembles',
            input=['mmff_opt_ensemble_xyz', 'orderfix_nonzero_testmols_sdf'], output='gfnff_postopt_ensemble_xyz', aware_keys=['method', 'timelimit'],
            pyfunction=gfnff_optimize_ensemble, calcdir='calcdir', nproc=maxproc,
            argv_prepare=lambda mmff_opt_ensemble_xyz, orderfix_nonzero_testmols_sdf, gfnff_postopt_ensemble_xyz, **kw:
                (mmff_opt_ensemble_xyz.access_element(), orderfix_nonzero_testmols_sdf.access_element(), gfnff_postopt_ensemble_xyz.get_path()),
            subs=lambda mmff_opt_ensemble_xyz, orderfix_nonzero_testmols_sdf, gfnff_postopt_ensemble_xyz, **kw: {
                'EXECSCRIPTS_DIR': execscripts_dir,
                'NUM_PROCS': maxproc,
                'start_xyz': mmff_opt_ensemble_xyz.access_element(),
                'initial_sdf': orderfix_nonzero_testmols_sdf.access_element(),
                'res_xyz': gfnff_postopt_ensemble_xyz.get_path(),
            },
            output_process=lambda gfnff_postopt_ensemble_xyz, **kw: confsearch.load_if_exists(gfnff_postopt_ensemble_xyz)
        ).greedy_on('gfnff_postopt_ensemble_xyz'),
        templates.exec('merge_optimized_ensembles',
            input=['mmff_opt_ensemble_xyz', 'gfnff_opt_ensemble_xyz', 'gfnff_postopt_ensemble_xyz'],
            output='opt_ensemble_xyz', merged_keys=['method', 'timelimit', 'testset', 'testcase'],
            method=lambda mmff_opt_ensemble_xyz, gfnff_opt_ensemble_xyz, gfnff_postopt_ensemble_xyz, opt_ensemble_xyz:
                merge_optimized_ensembles(
                    opt_ensemble_xyz,
                    mmff=mmff_opt_ensemble_xyz,
                    gfnff=gfnff_opt_ensemble_xyz,
                    gfnffpost=gfnff_postopt_ensemble_xyz,
                )
        ),
        
        templates.restrict('get_nonzero_summaries',
            input='final_summary_object', ref='opt_ensemble_xyz', output='nonzero_summary_object',
            merged_keys=['method', 'timelimit', 'testset', 'testcase', 'opttype'],
            ensure_one_to_one=False
        ),
        templates.restrict('get_postopt_nonzero_testmols',
            input='nonzero_testmols_sdf', ref='opt_ensemble_xyz', output='postopt_nonzero_testmols_sdf',
            merged_keys=['method', 'timelimit', 'testset', 'testcase', 'opttype'],
            ensure_one_to_one=False
        ),
        templates.pyfunction_subprocess('filter_duplicates',
            input=['opt_ensemble_xyz', 'nonzero_summary_object', 'postopt_nonzero_testmols_sdf'], output='filtered_ensemble_xyz',
            aware_keys=['method', 'opttype'],
            pyfunction=filter_ensembles, calcdir='calcdir', nproc=1, 
            argv_prepare=lambda opt_ensemble_xyz, nonzero_summary_object, postopt_nonzero_testmols_sdf, filtered_ensemble_xyz, **kw: (
                opt_ensemble_xyz.access_element(),
                nonzero_summary_object.access_element()['num_automorphisms'],
                filtered_ensemble_xyz.get_path(),
                postopt_nonzero_testmols_sdf.access_element()
            ),
            subs=lambda opt_ensemble_xyz, filtered_ensemble_xyz, postopt_nonzero_testmols_sdf, **kw: {
                'PYXYZ_SETTINGS': PYXYZ_SETTINGS,
                'USE_ENERGY_CUTOFF': kw['method'] in VERY_LARGE_ENSEMBLES_METHODS,
                # Replicate inputs/outputs for easier tempdir tracking
                'optimized_xyz': opt_ensemble_xyz.access_element(),
                'filtered_xyz': filtered_ensemble_xyz.get_path(),
                'topology_sdf': postopt_nonzero_testmols_sdf.access_element(),
            },
            output_process=lambda filtered_ensemble_xyz, **kw: confsearch.load_if_exists(filtered_ensemble_xyz)
        ).greedy_on('filtered_ensemble_xyz'),
    ]

    return benchmark_run_transforms
