import shutil
import json
import itertools

from pysquared import Transform
import pysquared.transforms.transform_templates as templates

from .timings import METHOD_NAMES
from .benchmark_run import mmff_optimize_ensemble, gfnff_optimize_ensemble
from utils import confsearch


def energies_dataitems():
    return {
        # Experimental conformers as XYZ
        'relevant_optimized_testmols_sdfs': {'type': 'file', 'mask': './energy_distribution/raw_experimental_optimize/relevant_opt_sdfs/{testset}_{testcase}.xyz'},
        'experimental_xyz': {'type': 'file', 'mask': './energy_distribution/raw_experimental_optimize/target_geoms/{testset}_{testcase}.xyz'},

        # Optimized experimental conformations containing their energies in description
        'mmff_experimental_opt_xyz': {'type': 'file', 'mask': './energy_distribution/raw_experimental_optimize/mmff/{testset}_{testcase}.xyz'},
        'gfnff_experimental_opt_xyz': {'type': 'file', 'mask': './energy_distribution/raw_experimental_optimize/gfnff/{testset}_{testcase}.xyz'},
        'gfnffpost_experimental_opt_xyz': {'type': 'file', 'mask': './energy_distribution/raw_experimental_optimize/gfnffpost/{testset}_{testcase}.xyz'},
        'opt_experimental_conformers':{'type': 'file', 'mask': './energy_distribution/experimental_optimize/{opttype}/{testset}_{testcase}.xyz'},
        'experimental_energy': {'type': 'object', 'keys': ['testset', 'testcase', 'opttype']},
        'experimental_energies_json': {'type': 'file', 'mask': './energy_distribution/dataframes/expenergies_{testset}.json'},

        # Compute relenergies of expconformers and generated ensembles
        'ensemble_relenergies_json': {'type': 'file', 'mask': './energy_distribution/relenergies/{relativity}_{level}/ensembles/{method}_{timelimit}/{testset}_{testcase}.json'},
        'expconformer_relenergies_json': {'type': 'file', 'mask': './energy_distribution/relenergies/{relativity}_{level}/experimental/{testset}_{testcase}.json'},
        'expconformer_relenergies': {'type': 'object', 'keys': ['relativity', 'level', 'testset', 'testcase']},
        'relativity_minima_json': {'type': 'file', 'mask': './energy_distribution/relenergies/{relativity}_{level}/minima/{testset}_{testcase}.json'},
        'final_ensemble_path': {'type': 'object', 'keys': ['relativity', 'level', 'method', 'timelimit', 'testset', 'testcase']},
        'final_expconformers_path': {'type': 'object', 'keys': ['relativity', 'level', 'testset', 'testcase']},
        
        # Build summary
        'csrun_stats_single_obj': {'type': 'object', 'keys': ['method', 'timelimit', 'testset', 'testcase']},
        'csrun_stats_single_obj_restricted': {'type': 'object', 'keys': ['relativity', 'level', 'method', 'timelimit', 'testset', 'testcase']},
        'total_single_csdata_json': {'type': 'file', 'mask': './energy_distribution/relenergies/{relativity}_{level}/ensemble_stats/{method}_{timelimit}/{testset}_{testcase}.json'},
        'total_single_csdata_obj': {'type': 'object', 'keys': ['relativity', 'level', 'testset', 'testcase', 'method', 'timelimit']},
    }

EXPXYZ_OPTTYPE_MAPS = {
    'mmff': 'mmff_experimental_opt_xyz',
    'gfnff': 'gfnff_experimental_opt_xyz',
    'gfnffpost': 'gfnffpost_experimental_opt_xyz',
}

def load_optimized_expconformers(opt_experimental_conformers, **input_items) -> None:
    from ringo import Confpool
    for opttype, itemname in EXPXYZ_OPTTYPE_MAPS.items():
        item = input_items[itemname]
        for xyz_name, keys in item:
            p = Confpool()
            p.include_from_file(xyz_name)
            assert len(p) == 1
            element_keys = {'opttype': opttype, **keys}
            new_xyz_path = opt_experimental_conformers.get_path(**element_keys)
            shutil.copy2(xyz_name, new_xyz_path)
            opt_experimental_conformers.include_element(new_xyz_path, **element_keys)


def extract_experimental_energies(xyz_name: str) -> float:
    from ringo import Confpool
    p = Confpool()
    p.include_from_file(xyz_name)
    assert len(p) == 1
    assert confsearch.ConformerInfo(description=p[0].descr).data['name'] == 'Experimental conformer'
    energy = confsearch.ConformerInfo(description=p[0].descr).energy
    return energy


def dump_expenergies_to_json(experimental_energy, experimental_energies_json) -> None:
    data = {}
    for energy, keys in experimental_energy:
        testcase = keys['testcase']
        if testcase not in data:
            data[testcase] = {}
        data[testcase][keys['opttype']] = energy
    
    result_json = experimental_energies_json.get_path()
    with open(result_json, 'w') as f:
        json.dump(data, f)
    experimental_energies_json.include_element(result_json)


def ensemble_to_level_basic(opttype, method, timelimit) -> str | None:
    requirements: list[bool] = [
        timelimit in (600, 'long'),
        method in ('ringo', 'ETKDGv3-2024', 'mtd', 'mmbasicOld', 'mmringOld', 'crestOld'),
        opttype == 'mmff',
    ]
    if not all(requirements):
        return None
    else:
        return 'mmff'
    
def ensemble_to_level_alllimit(opttype, method, timelimit) -> str | None:
    requirements: list[bool] = [
        # timelimit in (600, 'long'),
        method in ('ringo', 'ETKDGv3-2024', 'mtd', 'mmbasicOld', 'mmringOld', 'crestOld'),
        opttype == 'mmff',
    ]
    if not all(requirements):
        return None
    else:
        return 'mmff'

def ensemble_to_level_global(opttype, method, timelimit) -> str | None:
    requirements: list[bool] = [
        timelimit in (600, 'long'),
        method in METHOD_NAMES,
        opttype in ('mmff', 'gfnff', 'gfnffpost'),
    ]
    if not all(requirements):
        return None
    elif opttype == 'mmff':
        return 'mmff'
    elif method == 'ringo-vs-crest' and opttype == 'gfnffpost':
        return 'gfnff'
    elif method == 'crestOld' and opttype == 'gfnff':
        return 'gfnff'
    else:
        return None

def expenergy_to_level_basic(opttype) -> str | None:
    if opttype == 'mmff':
        return 'mmff'
    else:
        return None
    
def expenergy_to_level_global(opttype) -> str | None:
    if opttype == 'mmff':
        return 'mmff'
    elif opttype == 'gfnff':
        return 'gfnff'
    else:
        return None

ENERGY_RELATIVITIES = {
    'alllimit': {
        # MMFF only
        'ensemble_to_level': ensemble_to_level_alllimit,
        'expenergy_to_level': expenergy_to_level_basic,
    },
    'basic': {
        # MMFF only
        'ensemble_to_level': ensemble_to_level_basic,
        'expenergy_to_level': expenergy_to_level_basic,
    },
    'global': {
        # MMFF for all, GFNFF for CREST, GFNFF-post for MCR
        'ensemble_to_level': ensemble_to_level_global,
        'expenergy_to_level': expenergy_to_level_global,
    },
}


def add_relenergies_prep_subs(
    input_ensembles, # {method}, {timelimit}, {opttype}
    input_expenergies, # {opttype}
    res_ensemble_relenergies, # {relativity}, {level}, {method}, {timelimit}
    res_experimental_relenergies, # {relativity}, {level}
    res_minenergies, # {relativity}, {level}
) -> dict[str, list[dict[str, str]]]:
    (
        res_ensemble_relenergies_paths, res_experimental_relenergies_paths, res_minenergies_paths
    ) = generate_expected_relenergies_paths(
        input_ensembles, input_expenergies, res_ensemble_relenergies, res_experimental_relenergies, res_minenergies
    )
    res_subs = {
        'input_ensembles': [
            (path, keys)
            for path, keys in input_ensembles
        ],
        'absolute_expenergies': {
            keys['opttype']: energy
            for energy, keys in input_expenergies
        },
        'ensemble_relenergies_paths': res_ensemble_relenergies_paths,
        'experimental_relenergies_paths': res_experimental_relenergies_paths,
        'minenergies_paths': res_minenergies_paths,
    }
    return res_subs

def generate_expected_relenergies_paths(
    input_ensembles, # {method}, {timelimit}, {opttype}
    input_expenergies, # {opttype}
    res_ensemble_relenergies, # {relativity}, {level}, {method}, {timelimit}
    res_experimental_relenergies, # {relativity}, {level}
    res_minenergies, # {relativity}, {level}
):
    res_ensemble_relenergies_paths = {}
    res_experimental_relenergies_paths = {}
    res_minenergies_paths = {}
    for relativity, settings in ENERGY_RELATIVITIES.items():
        for path, keys in input_ensembles:
            level = settings['ensemble_to_level'](**keys)
            if level is None:
                continue
            rl_keys = {'relativity': relativity, 'level': level}
            res_minenergies_paths[relativity, level] = res_minenergies.get_path(**rl_keys)

            method, timelimit = keys['method'], keys['timelimit']
            res_ensemble_relenergies_paths[
                relativity, level, method, timelimit
            ] = res_ensemble_relenergies.get_path(
                **rl_keys, method=method, timelimit=timelimit
            )
        for energy, keys in input_expenergies:
            level = settings['expenergy_to_level'](**keys)
            if level is None:
                continue
            rl_keys = {'relativity': relativity, 'level': level}
            res_minenergies_paths[relativity, level] = res_minenergies.get_path(**rl_keys)
            res_experimental_relenergies_paths[relativity, level] = res_experimental_relenergies.get_path(**rl_keys)
    return res_ensemble_relenergies_paths, res_experimental_relenergies_paths, res_minenergies_paths


def add_relenergies_output_process(
    input_ensembles,
    input_expenergies,
    res_ensemble_relenergies,
    res_experimental_relenergies,
    res_minenergies
):
    res_ensemble_relenergies_paths, res_experimental_relenergies_paths, res_minenergies_paths = generate_expected_relenergies_paths(
        input_ensembles, input_expenergies, res_ensemble_relenergies, res_experimental_relenergies, res_minenergies
    )

    for item, paths in (
        (res_ensemble_relenergies, res_ensemble_relenergies_paths.values()),
        (res_experimental_relenergies, res_experimental_relenergies_paths.values()),
        (res_minenergies, res_minenergies_paths.values()),
    ):
        for path in paths:
            item.include_element(path)


def compile_ensemble_paths(ensembles, result_paths):
    for relativity, settings in ENERGY_RELATIVITIES.items():
        for path, keys in ensembles:
            level = settings['ensemble_to_level'](**keys)
            if level is not None:
                result_paths.include_element(
                    path,
                    relativity=relativity,
                    level=level,
                    method=keys['method'],
                    timelimit=keys['timelimit'],
                )

def compile_expconformer_paths(expconformers, result_paths):
    for relativity, settings in ENERGY_RELATIVITIES.items():
        for path, keys in expconformers:
            level = settings['expenergy_to_level'](**keys)
            if level is not None:
                result_paths.include_element(
                    path,
                    relativity=relativity,
                    level=level
                )


def record_relative_energies():
    import sys
    import os
    import json
    import itertools
    module_dir = INSERT_HERE
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    os.chdir(os.path.dirname(__file__))

    import ringo
    from pipelines.energies import ENERGY_RELATIVITIES
    import pandas as pd
    from utils.confsearch import ConformerInfo

    from icecream import install
    install()

    input_ensembles = INSERT_HERE
    absolute_expenergies = INSERT_HERE
    ensemble_relenergies_paths = INSERT_HERE
    experimental_relenergies_paths = INSERT_HERE
    minenergies_paths = INSERT_HERE

    ic(ensemble_relenergies_paths)
    ic(experimental_relenergies_paths)
    ic(minenergies_paths)

    for relativity, relativity_settings in ENERGY_RELATIVITIES.items():
        ensemble_df = {
            'conf_index': [],
            'method': [],
            'timelimit': [],
            'opttype': [],
            'level': [],
            'energy': [],
        }
        for xyzpath, keys in input_ensembles:
            level: str = relativity_settings['ensemble_to_level'](**keys)
            if level is None:
                continue

            p = ringo.Confpool()
            p.include_from_file(xyzpath)
            min_ener = None
            max_ener = None
            for i, m in enumerate(p):
                cur_energy: float = ConformerInfo(description=m.descr).energy
                ensemble_df['conf_index'].append(i)
                ensemble_df['method'].append(keys['method'])
                ensemble_df['timelimit'].append(keys['timelimit'])
                ensemble_df['opttype'].append(keys['opttype'])
                ensemble_df['level'].append(level)
                ensemble_df['energy'].append(cur_energy)
                if min_ener is None or cur_energy < min_ener:
                    min_ener = cur_energy
                if max_ener is None or cur_energy > max_ener:
                    max_ener = cur_energy
            ic(relativity, keys, xyzpath, level, min_ener, max_ener)
        ensemble_df = pd.DataFrame(ensemble_df)
        ic(ensemble_df)
        ensemble_df = ensemble_df.sort_values(by=['level']).reset_index(drop=True)
        ic(ensemble_df)
        ensemble_min_indices = ensemble_df.groupby('level')['energy'].idxmin()
        ensemble_min_rows = ensemble_df.loc[ensemble_min_indices]
        ic(ensemble_min_rows)

        expenergies_df = {
            'opttype': [],
            'level': [],
            'energy': [],
            'origin': [],
        }
        for opttype, cur_energy in absolute_expenergies.items():
            level: str = relativity_settings['expenergy_to_level'](opttype=opttype)
            if level is None:
                continue
            expenergies_df['opttype'].append(opttype)
            expenergies_df['level'].append(level)
            expenergies_df['energy'].append(cur_energy)
            expenergies_df['origin'].append('Experimental conformer')
        expenergies_df = pd.DataFrame(expenergies_df)

        levels = set(expenergies_df['level'].unique()) | set(ensemble_df['level'].unique())
        total_minimum_data: dict[str, dict | None] = {}
        for level in levels:
            total_minimum_data[level] = None
            ensemble_min_level = ensemble_min_rows[ensemble_min_rows['level'] == level]
            exp_min_level = expenergies_df[expenergies_df['level'] == level]
            assert len(ensemble_min_level) <= 1 and len(exp_min_level) <= 1
            assert len(ensemble_min_level) > 0 or len(exp_min_level) > 0

            for _, row in itertools.chain(ensemble_min_level.iterrows(), exp_min_level.iterrows()):
                cur_energy: float = row['energy']
                if total_minimum_data[level] is None or cur_energy < total_minimum_data[level]['energy']:
                    total_minimum_data[level] = row.to_dict()
            assert total_minimum_data[level] is not None
        
        ic(ensemble_df)
        ic(expenergies_df)
        calc_relenergy = lambda row: row['energy'] - total_minimum_data[row['level']]['energy']
        def all_positive_floats(value):
            return isinstance(value, float) and value >= 0
        
        if len(ensemble_df) > 0:
            ensemble_df['relenergy'] = ensemble_df.apply(calc_relenergy, axis=1)
            assert ensemble_df['relenergy'].apply(all_positive_floats).all()
        
        if len(expenergies_df) > 0:
            expenergies_df['relenergy'] = expenergies_df.apply(calc_relenergy, axis=1)
            assert expenergies_df['relenergy'].apply(all_positive_floats).all()
        ic(ensemble_df)
        ic(expenergies_df)

        for (cur_relativity, cur_level, cur_method, cur_timelimit), result_path in ensemble_relenergies_paths.items():
            if cur_relativity != relativity:
                continue
        
            relenergies_list = (
                ensemble_df[
                    (ensemble_df['level'] == cur_level) &
                    (ensemble_df['method'] == cur_method) &
                    (ensemble_df['timelimit'] == cur_timelimit)
                ]
                .sort_values(by='conf_index')
                ['relenergy']
                .to_list()
            )
            absenergies_list = (
                ensemble_df[
                    (ensemble_df['level'] == cur_level) &
                    (ensemble_df['method'] == cur_method) &
                    (ensemble_df['timelimit'] == cur_timelimit)
                ]
                .sort_values(by='conf_index')
                ['energy']
                .to_list()
            )
            
            with open(result_path, 'w') as f:
                json.dump({
                    'relative': relenergies_list,
                    'absolute': absenergies_list,
                }, f)
        
        for (cur_relativity, cur_level), result_path in experimental_relenergies_paths.items():
            if cur_relativity != relativity:
                continue
        
            cur_relenergies = (
                expenergies_df
                [expenergies_df['level'] == cur_level]
                ['relenergy']
                .to_list()
            )
            assert len(cur_relenergies) == 1
            
            cur_absenergies = (
                expenergies_df
                [expenergies_df['level'] == cur_level]
                ['energy']
                .to_list()
            )
            assert len(cur_absenergies) == 1

            with open(result_path, 'w') as f:
                json.dump({
                    'relative': cur_relenergies[0],
                    'absolute': cur_absenergies[0],
                }, f)

        for (cur_relativity, cur_level), result_path in minenergies_paths.items():
            if cur_relativity != relativity:
                continue

            cur_min_conformer_data = total_minimum_data[cur_level]
            ic(cur_min_conformer_data)
            with open(result_path, 'w') as f:
                json.dump(cur_min_conformer_data, f)


def augment_single_summary():
    import sys
    import os
    import json
    module_dir = INSERT_HERE
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    os.chdir(os.path.dirname(__file__))

    import ringo
    from utils.confsearch import ConformerInfo

    ensemble_xyz = INSERT_HERE
    raw_stats = INSERT_HERE
    relenergies_json_path = INSERT_HERE
    result_json_path = INSERT_HERE
    ENERGY_THRESHOLDS = INSERT_HERE

    import ringo
    from utils.confsearch import ConformerInfo

    with open(relenergies_json_path, 'r') as f:
        energies_data = json.load(f)
    absenergies = energies_data['absolute']
    relenergies = energies_data['relative']
    
    p = ringo.Confpool()
    p.include_from_file(ensemble_xyz)
    assert len(p) == len(relenergies)
    assert all(
        abs(ConformerInfo(description=m.descr).energy - absenergy) < 1e-10
        for m, absenergy in zip(p, absenergies)
    )

    energy_threshold_key = lambda e_thr: f'above_relenergy_{e_thr}'
    additional_data = {
        energy_threshold_key(e_thr): 0
        for e_thr in ENERGY_THRESHOLDS
    }
    def increase_key(keyname: str) -> None:
        if keyname not in additional_data:
            additional_data[keyname] = 0
        additional_data[keyname] += 1
    for m, relenergy in zip(p, relenergies):
        conformer_info = ConformerInfo(description=m.descr)
        status: str = conformer_info.status_flag
        increase_key(status)

        if conformer_info.is_failed:
            continue
        
        for e_thr in ENERGY_THRESHOLDS:
            if relenergy > e_thr:
                increase_key(energy_threshold_key(e_thr))

    result = {
        **raw_stats,
        **{
            f'num_{status_value}': cur_counts
            for status_value, cur_counts in additional_data.items()
        }
    }
    with open(result_json_path, 'w') as f:
        json.dump(result, f)


def energies_transforms(ds, main_logger, execscripts_dir: str, maxproc: int=1) -> list[Transform]:
    ENERGY_THRESHOLDS = [5, 10, 15]

    energies_transforms: list[Transform] = [
        #
        # ENERGIES OF EXPERIMENTAL CONFORMERS
        #

        # Convert experimental SDFs to XYZ
        templates.map('experimental_conformers_convert_to_xyz',
            input='optimized_testmols_sdfs', output='experimental_xyz', aware_keys=['testset'],
            mapping=lambda optimized_testmols_sdfs, experimental_xyz, testset: confsearch.sdf_to_xyz(
                sdf_path=optimized_testmols_sdfs,
                xyz_path=experimental_xyz,
                description='Experimental conformer'
            ) if testset == 'macromodel' else None,
            include_none=False
        ),
        templates.restrict('restrict_optimized_testmols_sdfs',
            input='optimized_testmols_sdfs', ref='experimental_xyz',
            output='relevant_optimized_testmols_sdfs',
            merged_keys=['testset', 'testcase']
        ),

        # Optimize experimental conformers in required force field
        templates.pyfunction_subprocess('mmff_experimental_optimize_ensembles',
            input=['experimental_xyz', 'relevant_optimized_testmols_sdfs'], output='mmff_experimental_opt_xyz',
            pyfunction=mmff_optimize_ensemble, calcdir='calcdir', nproc=1,
            argv_prepare=lambda experimental_xyz, relevant_optimized_testmols_sdfs, mmff_experimental_opt_xyz, **kw:
                (experimental_xyz.access_element(), relevant_optimized_testmols_sdfs.access_element(), mmff_experimental_opt_xyz.get_path()),
            subs=lambda experimental_xyz, relevant_optimized_testmols_sdfs, mmff_experimental_opt_xyz, **kw: {
                'NUM_PROCS': 1,
                'start_xyz': experimental_xyz.access_element(),
                'initial_sdf': relevant_optimized_testmols_sdfs.access_element(),
                'res_xyz': mmff_experimental_opt_xyz.get_path(),
            },
            output_process=lambda mmff_experimental_opt_xyz, **kw: confsearch.load_if_exists(mmff_experimental_opt_xyz)
        ),
        templates.pyfunction_subprocess('gfnff_experimental_optimize_ensembles',
            input=['experimental_xyz', 'relevant_optimized_testmols_sdfs'], output='gfnff_experimental_opt_xyz',
            pyfunction=gfnff_optimize_ensemble, calcdir='calcdir', nproc=1,
            argv_prepare=lambda experimental_xyz, relevant_optimized_testmols_sdfs, gfnff_experimental_opt_xyz, **kw:
                (experimental_xyz.access_element(), relevant_optimized_testmols_sdfs.access_element(), gfnff_experimental_opt_xyz.get_path()),
            subs=lambda experimental_xyz, relevant_optimized_testmols_sdfs, gfnff_experimental_opt_xyz, **kw: {
                'EXECSCRIPTS_DIR': execscripts_dir,
                'NUM_PROCS': 1,
                'start_xyz': experimental_xyz.access_element(),
                'initial_sdf': relevant_optimized_testmols_sdfs.access_element(),
                'res_xyz': gfnff_experimental_opt_xyz.get_path(),
            },
            output_process=lambda gfnff_experimental_opt_xyz, **kw: confsearch.load_if_exists(gfnff_experimental_opt_xyz)
        ),
        templates.pyfunction_subprocess('gfnffpost_experimental_optimize_ensembles',
            input=['mmff_experimental_opt_xyz', 'relevant_optimized_testmols_sdfs'], output='gfnffpost_experimental_opt_xyz',
            pyfunction=gfnff_optimize_ensemble, calcdir='calcdir', nproc=1,
            argv_prepare=lambda mmff_experimental_opt_xyz, relevant_optimized_testmols_sdfs, gfnffpost_experimental_opt_xyz, **kw:
                (mmff_experimental_opt_xyz.access_element(), relevant_optimized_testmols_sdfs.access_element(), gfnffpost_experimental_opt_xyz.get_path()),
            subs=lambda mmff_experimental_opt_xyz, relevant_optimized_testmols_sdfs, gfnffpost_experimental_opt_xyz, **kw: {
                'EXECSCRIPTS_DIR': execscripts_dir,
                'NUM_PROCS': 1,
                'start_xyz': mmff_experimental_opt_xyz.access_element(),
                'initial_sdf': relevant_optimized_testmols_sdfs.access_element(),
                'res_xyz': gfnffpost_experimental_opt_xyz.get_path(),
            },
            output_process=lambda gfnffpost_experimental_opt_xyz, **kw: confsearch.load_if_exists(gfnffpost_experimental_opt_xyz)
        ),
        templates.exec('load_optimized_expconformers',
            input=['mmff_experimental_opt_xyz', 'gfnff_experimental_opt_xyz', 'gfnffpost_experimental_opt_xyz'],
            output='opt_experimental_conformers',
            method=load_optimized_expconformers, merged_keys=['testcase', 'testset']
        ),
        templates.map('extract_experimental_energies',
            input='opt_experimental_conformers', output='experimental_energy',
            mapping=lambda opt_experimental_conformers: extract_experimental_energies(xyz_name=opt_experimental_conformers)
        ),
    
        templates.exec('dump_expenergies_to_json',
            input='experimental_energy', output='experimental_energies_json',
            method=dump_expenergies_to_json, merged_keys=['testcase', 'opttype']
        ),
        templates.map('load_expenergies_from_json',
            input='experimental_energies_json', output='experimental_energy',
            mapping=lambda experimental_energies_json: (
                (energy, {
                    'testcase': testcase,
                    'opttype': opttype,
                })
                for testcase, case_data in json.load(open(experimental_energies_json)).items()
                for opttype, energy in case_data.items()
            )
        ),

        #
        # BUILD SUMMARIES WITH RELATIVE ENERGIES
        #

        templates.pyfunction_subprocess('record_relative_energies',
            input=['filtered_ensemble_xyz', 'experimental_energy'],
            output=['ensemble_relenergies_json', 'expconformer_relenergies_json', 'relativity_minima_json'],
            aware_keys=['testset', 'testcase'], merged_keys=['method', 'timelimit', 'opttype'],
            pyfunction=record_relative_energies, calcdir='calcdir', nproc=1,
            argv_prepare=lambda testset, testcase, **kw: (testset, testcase),
            subs=lambda filtered_ensemble_xyz, experimental_energy,
                ensemble_relenergies_json, expconformer_relenergies_json, relativity_minima_json, **kw:
                add_relenergies_prep_subs(
                    input_ensembles=filtered_ensemble_xyz,
                    input_expenergies=experimental_energy,
                    res_ensemble_relenergies=ensemble_relenergies_json,
                    res_experimental_relenergies=expconformer_relenergies_json,
                    res_minenergies=relativity_minima_json,
                ),
            output_process=lambda filtered_ensemble_xyz, experimental_energy,
                ensemble_relenergies_json, expconformer_relenergies_json, relativity_minima_json, **kw:
                add_relenergies_output_process(
                    input_ensembles=filtered_ensemble_xyz,
                    input_expenergies=experimental_energy,
                    res_ensemble_relenergies=ensemble_relenergies_json,
                    res_experimental_relenergies=expconformer_relenergies_json,
                    res_minenergies=relativity_minima_json
                )
        ),
        templates.exec('compile_ensemble_paths_by_levels',
            input='filtered_ensemble_xyz', output='final_ensemble_path', merged_keys=['method', 'timelimit', 'opttype'],
            method=lambda filtered_ensemble_xyz, final_ensemble_path, **kw:
                compile_ensemble_paths(
                    ensembles=filtered_ensemble_xyz,
                    result_paths=final_ensemble_path,
                )
        ),
        templates.exec('compile_expconformer_paths_by_levels',
            input='opt_experimental_conformers', output='final_expconformers_path', merged_keys=['opttype'],
            method=lambda opt_experimental_conformers, final_expconformers_path, **kw:
                compile_expconformer_paths(
                    expconformers=opt_experimental_conformers,
                    result_paths=final_expconformers_path,
                )
        ),

        templates.load_json('load_csrun_stats', input='csrun_stats_single', output='csrun_stats_single_obj'),
        templates.restrict('restrict_csrun_stats_single_obj',
            input='csrun_stats_single_obj', ref='ensemble_relenergies_json', output='csrun_stats_single_obj_restricted',
            merged_keys=['relativity', 'level', 'testset', 'testcase', 'method', 'timelimit'],
            ensure_one_to_one=False
        ),
        templates.pyfunction_subprocess('augment_single_summary',
            input=['csrun_stats_single_obj_restricted', 'ensemble_relenergies_json', 'final_ensemble_path'], output='total_single_csdata_json',
            pyfunction=augment_single_summary, calcdir='calcdir', nproc=1,
            argv_prepare=lambda final_ensemble_path, ensemble_relenergies_json, total_single_csdata_json, **kw:
                (final_ensemble_path.access_element(), ensemble_relenergies_json.access_element(), total_single_csdata_json.get_path()),
            subs=lambda final_ensemble_path, csrun_stats_single_obj_restricted, ensemble_relenergies_json, total_single_csdata_json, **kw: {
                'raw_stats': csrun_stats_single_obj_restricted.access_element(),
                'ensemble_xyz': final_ensemble_path.access_element(),
                'relenergies_json_path': ensemble_relenergies_json.access_element(),
                'result_json_path': total_single_csdata_json.get_path(),
                'ENERGY_THRESHOLDS': ENERGY_THRESHOLDS,
            },
            output_process=lambda total_single_csdata_json, **kw: confsearch.assertive_include(total_single_csdata_json)
        ).greedy_on('total_single_csdata_json'),
        templates.load_json('load_total_single_stats',
            input='total_single_csdata_json', output='total_single_csdata_obj'
        ),
        templates.load_json('load_expconformer_relenergies',
            input='expconformer_relenergies_json', output='expconformer_relenergies',
            post_mapping=lambda obj, keys: (obj['relative'], keys)
        ),
    ]
    return energies_transforms
