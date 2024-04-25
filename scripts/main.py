# Do 'mamba activate mcrtest'

import os
import sys
import json
import time
import shutil
import io
import subprocess
import numpy as np

from icecream import install
install()

from pysquared import Transformator
from pysquared import DataStorage, Workflow, ThreadManager
import pysquared.transforms.transform_templates as templates
from pysquared import InstructionFactories as instr
from pysquared import execute_with_ui, create_logger

from utils.confsearch import assertive_include
from pipelines import (
    birdparser_dataitems, birdparser_transformator,
    quicksummary_dataitems, quicksummary_transforms,
    benchmark_dataitems, benchmark_transforms,
    energies_dataitems, energies_transforms,
    timings_dataitems, timings_transforms,
    expcompare_dataitems, expcompare_transforms,
    energy_distribution_dataitems, energy_distribution_transforms,
    diversity_dataitems, diversity_transforms,
    fullreport_dataitems, fullreport_transforms,
)

from chemscripts.utils import add_directory_to_path

import pandas as pd


class ArbitraryShrinkController:
    def __init__(self, keyname: str, max_allowed: int):
        self.allowed_values = []
        self.keyname = keyname
        self.max_allowed = max_allowed
    
    def check_keys(self, keys: dict) -> bool:
        if self.keyname not in keys or keys[self.keyname] in self.allowed_values:
            return True
        elif len(self.allowed_values) < self.max_allowed:
            self.allowed_values.append(keys[self.keyname])
            return True
        else:
            return False

# Generated using utils/split_macromodel.py
# CHECK that splitting is correct
# MACROMODEL_TESTSET_SPLIT = {
#     'A': ['pdb2VYP', 'pdb2C7X', 'pdb1YND', 'csdRUQVAB', 'pdb2PH8', 'csdKAQJAN', 'pdb1XS7', 'pdb1NMK', 'csdDACMAW', 'pdb2ASP', 'csdNUCZUH', 'csdKAQJER', 'csdRASTIO', 'pdb1FKD', 'pdb2WHW', 'pdb2F3F', 'csdXAGXOT', 'csdVAKPIH', 'pdb3I6O', 'pdb1PPJ', 'csdDEKSAN', 'pdb1NT1', 'pdb1NSG', 'pdb1MRL', 'csdGEZBUI', 'pdb2F3E', 'pdb2DG4', 'pdb2XBK', 'csdCGPGAP10', 'csdGIHKOX10', 'csdWENFOL', 'csdXACJAM', 'pdb1YET', 'pdb1QZ5', 'csdMUCYIT', 'csdRECRAT', 'csdMIWTER', 'pdb2C6H'],
#     'B': ['csdCTVHVH01', 'csdCIYLAY', 'csdYAGKOG', 'csdPAPGAP', 'csdHEBLIJ', 'pdb1NTK', 'csdKIVDIC', 'pdb1PFE', 'csdQOSNAN', 'pdb2VZM', 'csdHAXMOI10', 'csdBIHYAS10', 'csdCHPSAR', 'pdb2WEA', 'csdCALSAR', 'pdb2OTJ', 'csdAAGGAG10', 'csdCUQRUB', 'csdZAJDUJ', 'csdFIXYEQ', 'pdb3BXR', 'csdXAGYAG', 'csdWUMKIY', 'csdCLPGDH', 'csdMALVAL10', 'csdXAGXIN', 'pdb3ABA', 'csdCIYKUR', 'csdWUBGAB', 'pdb2ASO', 'pdb1S22', 'csdHIKPAS01', 'pdb3K5C', 'csdYUPBAN01', 'pdb1BZL', 'csdRULSUN', 'csdYIVNOG', 'csdFINWEE10'],
#     'C': ['pdb1PKF', 'csdOHIDUE', 'pdb3KEE', 'pdb2ASM', 'csdVALINM30', 'csdFEGJAD', 'pdb1JFF', 'csdCAZVEF', 'csdTERDAW', 'csdOHIFAM', 'pdb1KEG', 'csdVOKBIG', 'csdCAMVES01', 'pdb1WUA', 'csdLALWIF', 'csdABOMOT', 'pdb1QZ6', 'pdb1TPS', 'csdYAGVEH', 'pdb3DV5', 'pdb1EHL', 'csdVOLHOU', 'pdb2GPL', 'csdIDIMAJ', 'pdb1GHG', 'pdb1SKX', 'csdJOKSAD', 'csdFIVSAE', 'csdFAGFEZ', 'pdb1KHR', 'csdYUYYEW', 'csdQEKNID', 'pdb1E9W', 'pdb1YXQ', 'csdCTILIH10', 'pdb3M6G', 'pdb2IYA', 'pdb1NWX'],
#     'D': ['csdCYVIHV10', 'csdGAFSEL', 'csdQUDROW', 'pdb2IYF', 'csdDALPUC', 'pdb1QPL', 'csdPOXTRD10', 'csdCAHWEN', 'csdOHUXEU', 'pdb1XBP', 'csdJINWUZ', 'csdABOJOR', 'pdb1FKI', 'csdNITFEB', 'csdMECBOL', 'pdb2Q0R', 'csdACICOE', 'csdCOHVAW', 'pdb1FKJ', 'pdb7UPJ', 'pdb3DV1', 'csdDIGTOC', 'pdb1ESV', 'pdb1A7X', 'csdICYSPA', 'pdb1QPF', 'csdREMCOC', 'pdb3EKS', 'csdVUHHEM', 'csdQUDRIQ', 'pdb1FKL', 'csdBIDPIN10', 'csdDAFGIA', 'pdb1BXO', 'pdb1OSF', 'pdb2QZK']
# }
# MACROMODEL_TESTSET_SPLIT = {
#     'A': ['csdRASTIO', 'csdDACMAW', 'csdCGPGAP10', 'csdGEZBUI', 'csdQUDRIQ', 'pdb1YXQ', 'csdYAGKOG', 'pdb1FKJ', 'pdb2VYP', 'pdb1NTK', 'pdb1ESV', 'pdb1YND', 'csdYAGVEH', 'csdCALSAR', 'csdCHPSAR', 'csdACICOE', 'pdb1BZL', 'pdb1PKF', 'csdPOXTRD10', 'pdb2OTJ', 'pdb1SKX', 'csdMALVAL10', 'csdDIGTOC', 'pdb1XS7', 'pdb3KEE', 'pdb1QZ6', 'csdTERDAW', 'csdIDIMAJ', 'pdb2GPL', 'pdb3ABA'], 
#     'B': ['csdFIVSAE', 'pdb2VZM', 'pdb2WHW', 'csdDAFGIA', 'csdMECBOL', 'csdCYVIHV10', 'csdICYSPA', 'pdb1NT1', 'csdOHIFAM', 'csdNITFEB', 'pdb1QPL', 'csdHEBLIJ', 'csdQOSNAN', 'csdCIYKUR', 'pdb2F3F', 'csdFINWEE10', 'csdXAGXOT', 'csdCAHWEN', 'pdb1GHG', 'pdb2C6H', 'pdb1QPF', 'pdb1YET', 'csdFIXYEQ', 'pdb2C7X', 'csdCIYLAY', 'pdb1JFF', 'csdPAPGAP', 'csdXAGXIN', 'pdb2ASP', 'csdCAZVEF'], 
#     'B2': ['pdb3EKS', 'csdBIHYAS10', 'pdb2ASO', 'csdOHUXEU', 'pdb1E9W', 'pdb1EHL', 'csdMIWTER', 'pdb3DV5', 'csdCLPGDH', 'pdb2IYA', 'pdb7UPJ', 'csdLALWIF', 'pdb1KEG', 'pdb2WEA', 'csdVAKPIH', 'pdb1S22', 'pdb2Q0R', 'csdYUPBAN01', 'csdKAQJAN', 'csdWENFOL', 'pdb1KHR', 'pdb2F3E', 'csdXAGYAG', 'csdNUCZUH', 'csdVUHHEM', 'csdBIDPIN10', 'pdb1A7X', 'csdDALPUC', 'pdb1MRL', 'pdb2XBK'], 
#     'E': ['csdMUCYIT', 'pdb3K5C', 'pdb2IYF', 'csdKIVDIC', 'csdAAGGAG10', 'csdQEKNID', 'pdb1FKI', 'csdCTILIH10', 'pdb1FKD', 'csdRECRAT', 'csdRUQVAB', 'pdb1TPS', 'pdb3M6G', 'csdVOLHOU', 'pdb1NMK', 'pdb1WUA', 'pdb1XBP', 'pdb1NSG', 'csdQUDROW', 'csdZAJDUJ', 'csdRULSUN', 'csdREMCOC', 'csdFEGJAD', 'csdKAQJER', 'pdb2ASM', 'csdCUQRUB', 'pdb2QZK', 'pdb1FKL', 'csdJINWUZ', 'pdb3DV1'], 
#     'F': ['csdCTVHVH01', 'pdb3BXR', 'csdWUMKIY', 'csdGAFSEL', 'csdABOMOT', 'csdYIVNOG', 'pdb1PFE', 'csdJOKSAD', 'csdCAMVES01', 'pdb1BXO', 'pdb3I6O', 'pdb1NWX', 'pdb2PH8', 'csdWUBGAB', 'pdb1QZ5', 'csdHAXMOI10', 'csdGIHKOX10', 'pdb1OSF', 'csdXACJAM', 'csdYUYYEW', 'csdDEKSAN', 'csdFAGFEZ', 'csdCOHVAW', 'pdb2DG4', 'pdb1PPJ', 'csdOHIDUE', 'csdHIKPAS01', 'csdVOKBIG', 'csdVALINM30', 'csdABOJOR'],
# }
# MACROMODEL_TESTSET_SPLIT = {
#     'A': ['csdCAZVEF', 'pdb2F3E', 'csdGEZBUI', 'pdb1SKX', 'pdb2F3F', 'csdCIYKUR', 'csdCAMVES01', 'pdb1WUA', 'pdb1NT1', 'pdb1YND', 'pdb1XBP', 'csdFAGFEZ', 'csdWUMKIY', 'csdPOXTRD10', 'pdb1QPF', 'csdKIVDIC', 'pdb1MRL', 'pdb1S22', 'csdCYVIHV10', 'csdBIHYAS10', 'pdb1QZ6', 'pdb1PFE', 'csdOHIFAM', 'pdb1NTK', 'csdQEKNID'], 
#     'B': ['csdMECBOL', 'pdb2XBK', 'csdXAGXOT', 'pdb2C7X', 'pdb1YET', 'csdXACJAM', 'pdb1BZL', 'csdIDIMAJ', 'csdQOSNAN', 'csdYAGVEH', 'csdCOHVAW', 'pdb1XS7', 'csdABOJOR', 'pdb1NWX', 'csdDAFGIA', 'pdb1A7X', 'csdCTILIH10', 'pdb1NMK', 'pdb1FKI', 'pdb1KEG', 'pdb3I6O', 'csdKAQJAN', 'csdWENFOL', 'csdACICOE', 'pdb1GHG'], 
#     'C': ['csdOHUXEU', 'csdAAGGAG10', 'csdREMCOC', 'pdb7UPJ', 'csdHIKPAS01', 'pdb2Q0R', 'pdb3ABA', 'csdVUHHEM', 'csdCALSAR', 'pdb2ASM', 'csdQUDRIQ', 'pdb1FKL', 'pdb1EHL', 'csdXAGXIN', 'pdb1KHR', 'pdb1E9W', 'csdJOKSAD', 'pdb1TPS', 'csdMUCYIT', 'pdb2IYA', 'pdb1JFF', 'pdb1PPJ', 'pdb2ASP', 'csdVOLHOU', 'csdVAKPIH'], 
#     'D': ['csdHEBLIJ', 'pdb2OTJ', 'pdb1PKF', 'csdCLPGDH', 'pdb3KEE', 'pdb2ASO', 'csdCHPSAR', 'csdXAGYAG', 'csdDIGTOC', 'pdb1FKJ', 'pdb3DV1', 'csdNITFEB', 'csdQUDROW', 'csdRULSUN', 'csdFINWEE10', 'csdVOKBIG', 'csdYAGKOG', 'csdRUQVAB', 'pdb3M6G', 'csdFIXYEQ', 'csdGIHKOX10', 'csdVALINM30', 'pdb2GPL', 'pdb2WHW', 'csdOHIDUE'], 
#     'E': ['pdb1BXO', 'csdCGPGAP10', 'csdKAQJER', 'pdb2IYF', 'pdb2VYP', 'csdRECRAT', 'csdLALWIF', 'csdFEGJAD', 'csdDACMAW', 'csdCIYLAY', 'csdPAPGAP', 'pdb2DG4', 'csdJINWUZ', 'pdb1NSG', 'csdZAJDUJ', 'csdYUPBAN01', 'csdFIVSAE', 'pdb1QZ5', 'csdYIVNOG', 'csdRASTIO', 'pdb2VZM', 'csdCTVHVH01', 'csdNUCZUH', 'pdb3K5C', 'pdb1FKD'], 
#     'F': ['pdb2QZK', 'csdCAHWEN', 'csdWUBGAB', 'csdDEKSAN', 'csdMALVAL10', 'csdBIDPIN10', 'csdTERDAW', 'pdb2WEA', 'csdICYSPA', 'csdCUQRUB', 'pdb2C6H', 'pdb1ESV', 'csdDALPUC', 'pdb3DV5', 'pdb3BXR', 'csdMIWTER', 'csdABOMOT', 'pdb1QPL', 'pdb1YXQ', 'pdb3EKS', 'pdb1OSF', 'csdGAFSEL', 'pdb2PH8', 'csdYUYYEW', 'csdHAXMOI10']
# }
# To redo all MMFF calcs by fixing energies
# MACROMODEL_TESTSET_SPLIT = {
#     'C': ['pdb2VYP', 'pdb2C7X', 'pdb1YND', 'csdRUQVAB', 'pdb2PH8', 'csdKAQJAN', 'pdb1XS7', 'pdb1NMK', 'csdDACMAW', 'pdb2ASP', 'csdNUCZUH', 'csdKAQJER', 'csdRASTIO', 'pdb1FKD', 'pdb2WHW', 'pdb2F3F', 'csdXAGXOT', 'csdVAKPIH', 'pdb3I6O', 'pdb1PPJ', 'csdDEKSAN', 'pdb1NT1', 'pdb1NSG', 'pdb1MRL', 'csdGEZBUI', 'pdb2F3E', 'pdb2DG4', 'pdb2XBK', 'csdCGPGAP10', 'csdGIHKOX10', 'csdWENFOL', 'csdXACJAM', 'pdb1YET', 'pdb1QZ5', 'csdMUCYIT', 'csdRECRAT', 'csdMIWTER', 'pdb2C6H', 'csdCTVHVH01', 'csdCIYLAY', 'csdYAGKOG', 'csdPAPGAP', 'csdHEBLIJ', 'pdb1NTK', 'csdKIVDIC', 'pdb1PFE', 'csdQOSNAN', 'pdb2VZM', 'csdHAXMOI10', 'csdBIHYAS10', 'csdCHPSAR', 'pdb2WEA', 'csdCALSAR', 'pdb2OTJ', 'csdAAGGAG10', 'csdCUQRUB', 'csdZAJDUJ', 'csdFIXYEQ', 'pdb3BXR', 'csdXAGYAG', 'csdWUMKIY', 'csdCLPGDH', 'csdMALVAL10', 'csdXAGXIN', 'pdb3ABA', 'csdCIYKUR', 'csdWUBGAB', 'pdb2ASO', 'pdb1S22', 'csdHIKPAS01', 'pdb3K5C', 'csdYUPBAN01', 'pdb1BZL', 'csdRULSUN', 'csdYIVNOG', 'csdFINWEE10'],
#     'F': ['pdb1PKF', 'csdOHIDUE', 'pdb3KEE', 'pdb2ASM', 'csdVALINM30', 'csdFEGJAD', 'pdb1JFF', 'csdCAZVEF', 'csdTERDAW', 'csdOHIFAM', 'pdb1KEG', 'csdVOKBIG', 'csdCAMVES01', 'pdb1WUA', 'csdLALWIF', 'csdABOMOT', 'pdb1QZ6', 'pdb1TPS', 'csdYAGVEH', 'pdb3DV5', 'pdb1EHL', 'csdVOLHOU', 'pdb2GPL', 'csdIDIMAJ', 'pdb1GHG', 'pdb1SKX', 'csdJOKSAD', 'csdFIVSAE', 'csdFAGFEZ', 'pdb1KHR', 'csdYUYYEW', 'csdQEKNID', 'pdb1E9W', 'pdb1YXQ', 'csdCTILIH10', 'pdb3M6G', 'pdb2IYA', 'pdb1NWX', 'csdCYVIHV10', 'csdGAFSEL', 'csdQUDROW', 'pdb2IYF', 'csdDALPUC', 'pdb1QPL', 'csdPOXTRD10', 'csdCAHWEN', 'csdOHUXEU', 'pdb1XBP', 'csdJINWUZ', 'csdABOJOR', 'pdb1FKI', 'csdNITFEB', 'csdMECBOL', 'pdb2Q0R', 'csdACICOE', 'csdCOHVAW', 'pdb1FKJ', 'pdb7UPJ', 'pdb3DV1', 'csdDIGTOC', 'pdb1ESV', 'pdb1A7X', 'csdICYSPA', 'pdb1QPF', 'csdREMCOC', 'pdb3EKS', 'csdVUHHEM', 'csdQUDRIQ', 'pdb1FKL', 'csdBIDPIN10', 'csdDAFGIA', 'pdb1BXO', 'pdb1OSF', 'pdb2QZK']
# }


# All heavy molecules
# MACROMODEL_TESTSET_SPLIT = {
#     'A2': ['pdb2IYA', 'csdMIWTER', 'csdRECRAT'],
#     'B2': ['pdb1NWX', 'pdb3M6G'],
#     'C2': ['csdYIVNOG', 'pdb2C6H', 'pdb2QZK'],
#     'D2': ['csdFINWEE10', 'csdRULSUN'],
# }
        
# Rushing to finish MMFF
# MACROMODEL_TESTSET_SPLIT = { 'A': ['csdFINWEE10'], 'B': ['csdRULSUN'], 'D': ['pdb1NWX'], 'E': ['pdb3M6G'],}


PROJECT_DIR = '../project_data'
EXECSCRIPTS_DIR = './execscripts'
# TEMPFILES_DIR = '../tempdirs' # CHECK Worker job

IN_RAM_TEMPDIR = True # CHECK Whether in-ram storage is desired

# This one will be passed to subprocesses
EXECSCRIPTS_DIR = os.path.abspath(EXECSCRIPTS_DIR)

def main(stack_stream=None, prompt_transformator_path=False):
    RUNNER_NODE = sys.argv[1]
    # CURRENT_TESTCASES = MACROMODEL_TESTSET_SPLIT[RUNNER_NODE]
    # assert RUNNER_NODE in MACROMODEL_TESTSET_SPLIT, f"Node '{RUNNER_NODE}' is unknown"

    #
    # UNIVERSAL SETTINGS
    #
    if IN_RAM_TEMPDIR:
        tempdir_root = '/dev/shm'
    else:
        tempdir_root = '.'

    # CHECK Correct line uncommented
    # thread_manager = ThreadManager(
    #     wd=TEMPFILES_DIR,
    #     maxproc=46
    # )
    thread_manager = ThreadManager(
        wd=os.path.join(tempdir_root, f'tempfiles_{RUNNER_NODE}'),
        maxproc=54
    )

    main_logger = create_logger("Main", filename=f'{RUNNER_NODE}.log') # CHECK Log-file name

    add_directory_to_path(EXECSCRIPTS_DIR)

    ds = DataStorage({
            **birdparser_dataitems(),
            **quicksummary_dataitems(),
            **benchmark_dataitems(),
            **energies_dataitems(),
            **timings_dataitems(),
            **energy_distribution_dataitems(),
            **diversity_dataitems(),
            **expcompare_dataitems(),
            **fullreport_dataitems(),
        },
        # logger=main_logger,
        allow_overwrite=True,
        allow_lazy_checkin=True,
        wd=PROJECT_DIR
    )

    # bird_parser = birdparser_transformator(ds, main_logger)
    main_benchmark_transformator = Transformator(transformations=[
            *quicksummary_transforms(ds, main_logger),
            *benchmark_transforms(ds, main_logger, maxproc=thread_manager.maxproc, execscripts_dir=EXECSCRIPTS_DIR),
            *energies_transforms(ds, main_logger, maxproc=thread_manager.maxproc, execscripts_dir=EXECSCRIPTS_DIR),
            *timings_transforms(ds, main_logger),
            *expcompare_transforms(ds, main_logger),
            *energy_distribution_transforms(ds, main_logger, maxproc=thread_manager.maxproc, execscripts_dir=EXECSCRIPTS_DIR),
            *diversity_transforms(ds, main_logger, maxproc=thread_manager.maxproc),
            *fullreport_transforms(ds, main_logger),
        ],
        storage=ds,
        logger=main_logger,
    )

    workflow = Workflow(
        transformators={
            # 'bird_parser': bird_parser,
            'main_benchmark': main_benchmark_transformator,
        },
        storage=ds,
        logger=main_logger,
    )

    testcase_controller = ArbitraryShrinkController(keyname='testcase', max_allowed=2)
    method_controller = ArbitraryShrinkController(keyname='method', max_allowed=1)

    def benchmark_reach(target: str, **kw):
        return instr.reach_target(transformator='main_benchmark', target=target, **kw)


    # CHECK Correct target(s) are selected
    workflow.execute([
        # instr.reach_target(transformator='bird_parser', target='full_testset_summary'),
        # instr.reach_target(transformator='bird_parser', target='testset_summary'),
        # instr.reach_target(transformator='bird_parser', target='aug_testset_summary'),
        # instr.reach_target(transformator='bird_parser', target='randomized_testmols_sdfs', forward={'keys_control': lambda keys: True}),

        benchmark_reach(target='quick_testset_summary_excel'),
        benchmark_reach(target='final_testset_summary_excel'),
        benchmark_reach(target='final_summary_object', forward={'keys_control': lambda keys: True}),
        benchmark_reach(target='heavy_final_summary_object', forward={'keys_control': lambda keys: True}),
        
        # Sampling runs
        benchmark_reach(target='mcr_stats_single'),
        # benchmark_reach(target='mcrCN_stats_single'),
        benchmark_reach(target='mcr_vs_rdkit_stats_single'),
        benchmark_reach(target='mcr_vs_mtd_stats_single'),
        benchmark_reach(target='mcr_vs_crest_stats_single'),
        # benchmark_reach(target='mcrCN_vs_rdkit_stats_single'),
        # benchmark_reach(target='mcrCN_vs_mtd_stats_single'),
        # benchmark_reach(target='mcrCN_vs_crest_stats_single'),
        benchmark_reach(target='mtd_ensemble_xyz'),
        benchmark_reach(target='rdkitv1_stats_single'),
        benchmark_reach(target='rdkitv3_2024_stats_single'),
        benchmark_reach(target='mcr_vs_rdkit_ensemble_xyz'),
        benchmark_reach(target='mcr_vs_rdkit2024_stats_single'),
        benchmark_reach(target='mcr_vs_mmring_stats_single'),

        # Postprocessing
        benchmark_reach(target='raw_ensemble_xyz'),
        benchmark_reach(target='nonzero_testmols_sdf'),
        benchmark_reach(target='fixed_ordering_ensemble_xyz'),
        benchmark_reach(target='orderfix_nonzero_testmols_sdf'),
        benchmark_reach(target='mmff_opt_ensemble_xyz'),
        benchmark_reach(target='gfnff_opt_ensemble_xyz'),
        benchmark_reach(target='gfnff_postopt_ensemble_xyz'),
        benchmark_reach(target='opt_ensemble_xyz'),
        benchmark_reach(target='postopt_nonzero_testmols_sdf', forward={'keys_control': lambda keys: True}),
        benchmark_reach(target='nonzero_summary_object', forward={'keys_control': lambda keys: True}),
        benchmark_reach(target='filtered_ensemble_xyz'),
        benchmark_reach(target='ensemble_relenergies_json'),
        benchmark_reach(target='experimental_energies_json'),
        benchmark_reach(target='final_ensemble_path'),
        benchmark_reach(target='total_single_csdata_json'),

        benchmark_reach(target='timing_df'),
        benchmark_reach(target='timings_raw_csv'),
        benchmark_reach(target='timing_plot_settings', forward={'keys_control': lambda keys: True}),
        benchmark_reach(target='timing_df_with_data'),
        benchmark_reach(target='timing_df_processed'),
        benchmark_reach(target='timings_final_csv'),
        benchmark_reach(target='timings_plot_png'),

        benchmark_reach(target='energy_distribution_csv', forward={'keys_control': lambda keys: True}),
        benchmark_reach(target='energy_distr_plotting_data', forward={'keys_control': lambda keys: True}),
        benchmark_reach(target='edistr_expconformer_relenergies', forward={'keys_control': lambda keys: True}),
        benchmark_reach(target='energy_distr_single_plot'),
        benchmark_reach(target='esummary_typed_report_pdf'),

        benchmark_reach(target='diversity_ensemble_xyz_paths', forward={'keys_control': lambda keys: True, 'prompt_transformator_path': False}),
        benchmark_reach(target='ensemble_relenergies_json_for_diversity', forward={'keys_control': lambda keys: True, 'prompt_transformator_path': False}),
        benchmark_reach(target='initial_diversity_ensemble'),
        benchmark_reach(target='merged_diversity_ensemble'),
        benchmark_reach(target='crmsd_matrix'),
        benchmark_reach(target='clustering_indices_json'),
        benchmark_reach(target='bestclustering_indices_json'),
        benchmark_reach(target='clustered_crmsd_matrix_paths', forward={'keys_control': lambda keys: True, 'prompt_transformator_path': False}),
        benchmark_reach(target='cluster_medoid_indices_json'),
        benchmark_reach(target='embedding_2d_json'),
        benchmark_reach(target='clustered_final_ensemble_paths', forward={'keys_control': lambda keys: True}),
        benchmark_reach(target='point_df_csvs'),
        benchmark_reach(target='cluster_df_csvs'),
        benchmark_reach(target='timing_info'),
        benchmark_reach(target='diversity_df_csvs'),
        benchmark_reach(target='diversity_plot_png'),
        benchmark_reach(target='diversity_plot_svg'),
        benchmark_reach(target='diversity_summary_pdf'),
        benchmark_reach(target='diversity_crosscompare_data'),
        benchmark_reach(target='diversity_crosscompare_json'),
        benchmark_reach(target='diversity_crosscompare_results_json'),
        benchmark_reach(target='diversity_crosscompare_analysis_csv'),
        benchmark_reach(target='diversity_crosscompare_summary_csv'),

        benchmark_reach(target='ensembles_for_expcompare', forward={'keys_control': lambda keys: True}),
        benchmark_reach(target='expconformers_for_expcompare', forward={'keys_control': lambda keys: True}),
        benchmark_reach(target='exp_compare_result'),
        benchmark_reach(target='compare_df_csv'),
        benchmark_reach(target='compare_summary_csv'),
        benchmark_reach(target='compare_summary_plot_svg'),
        benchmark_reach(target='compare_summary_plot_png'),

        benchmark_reach(target='sampling_rates_dfs_csv'),
        benchmark_reach(target='energy_distr_features_dfs_csv'),
        benchmark_reach(target='expconformer_crmsd_dfs_csv'),
        benchmark_reach(target='energy_distr_plots_finalreport'),
        benchmark_reach(target='diversity_plot_finalreport'),
        benchmark_reach(target='finalreport_pdf'),
    ], forward={
        'logger': main_logger,
        'stack_stream': stack_stream,
        'thread_manager': thread_manager,
        'prompt_transformator_path': prompt_transformator_path,
        
        # CHECK Control for full testset
        # 'keys_control': lambda keys: ('testset' not in keys or keys['testset'] in ('macromodel',))
    })

    main_logger.warning('DONE')


if __name__ == "__main__":
    # execute_with_ui(main)
    # logger = create_logger("Main", filename='head.log')
    main(prompt_transformator_path=('-p' in sys.argv))

# Before running XTB
"""
mamba activate mcrtest
module load intel-parallel-studio/2017 gcc/10.2.0
"""
# CHECK All changes are saved
# CHECK Run 'module load intel-parallel-studio/2017 gcc/10.2.0' before XTB calls
# CHECK Run with either 'python main.py' or 'python main.py A'
