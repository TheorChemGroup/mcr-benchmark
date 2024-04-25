import shutil
import itertools
from typing import Literal, Optional

from pysquared import Transform
import pysquared.transforms.transform_templates as templates

from .bird_parser import get_basic_testset_summary, generic_summary_generation_blueprint
from utils import confsearch


def quicksummary_dataitems() -> dict[str, dict]:
    return {
        # Also need 'optimized_testmols_sdfs' which are loaded from bird_parser.py
        'quick_summary_object': {'type': 'object', 'keys': ['testset', 'testcase']},
        'quick_summary_block': {'type': 'object', 'keys': ['testset']},
        'quick_testset_summary_excel': {'type': 'file', 'mask': './testsets/{testset}/testset_summary_quick.xlsx'},
        'final_testset_summary_excel': {'type': 'file', 'mask': './testsets/{testset}/testset_summary_final.xlsx'},

        # These will be used later for loading 'final_testset_summary_excel'
        'final_summary_object': {'type': 'object', 'keys': ['testset', 'testcase']},
        'final_summary_block': {'type': 'object', 'keys': ['testset']},
    }


def quick_summary_generation_blueprint(
        grouped_transforms: dict[Literal['get', 'merge', 'save'], str],
        item_names: dict[Literal['sdf', 'summary', 'block', 'xlsx'], str | list[str]],
        testcase_key: str='testcase',
        main_block_name: str='Main'
    ) -> tuple[Transform]:
    other_item_names = {
        key: value
        for key, value in item_names.items()
        if key != 'sdf'
    }
    return generic_summary_generation_blueprint(
        item_names={
            'inputs': [item_names['sdf']],
            **other_item_names
        },
        summary_generator=lambda **kwargs: {
            **get_basic_testset_summary(
                sdf_name=kwargs[item_names['sdf']],
                testcase=kwargs[testcase_key]
            ),
            'num_automorphisms': confsearch.sdf_to_confpool(kwargs[item_names['sdf']]).get_num_isomorphisms(),
            'smiles': confsearch.sdf_to_smiles(kwargs[item_names['sdf']]),
        },
        grouped_transforms=grouped_transforms,
        testcase_key=testcase_key,
        main_block_name=main_block_name,   
    )


def excel_parsing_blueprint(
        item_names: dict[Literal['xlsx', 'block', 'object'], str | list[str]],
        grouped_transforms: Optional[dict[Literal['parse', 'extract'], str]] = None,
    ) -> tuple[Transform]:
    if grouped_transforms is None:
        grouped_transforms = {
            'parse': f"{item_names['xlsx']}->{item_names['block']}",
            'extract': f"{item_names['block']}->{item_names['object']}",
        }
    return (
        templates.parse_xlsx(grouped_transforms['parse'], input=item_names['xlsx'], output=item_names['block']),
        templates.extract_df_rows(grouped_transforms['extract'], input=item_names['block'], output=item_names['object']),
    )


def quicksummary_transforms(ds, main_logger) -> list[Transform]:
    def choose_appropriate_summary(quick_testset_summary_excel, aug_testset_summary, final_testset_summary_excel):
        quick_summaries = {
            keys['testset']: file
            for file, keys in quick_testset_summary_excel
        }
        full_summaries = {
            keys['testset']: file
            for file, keys in aug_testset_summary
        }
        testsets = set(
            testset
            for testset in itertools.chain(quick_summaries.keys(), full_summaries.keys())
        )

        for testset in testsets:
            target_path: str = final_testset_summary_excel.get_path(testset=testset)
            if testset in full_summaries:
                shutil.copy2(full_summaries[testset], target_path)
            else:
                shutil.copy2(quick_summaries[testset], target_path)
            final_testset_summary_excel.include_element(target_path, testset=testset)


    quicksummary_transforms: list[Transform] = [
        *quick_summary_generation_blueprint(
            grouped_transforms={
                'get': 'construct_quick_summary',
                'merge': 'merge_quick_summaries',
                'save': 'gen_quick_summary_excel',
            },
            item_names={
                'sdf': 'optimized_testmols_sdfs',
                'summary': 'quick_summary_object',
                'block': 'quick_summary_block',
                'xlsx': 'quick_testset_summary_excel',
            },
        ),
        templates.exec(
            'load_final_testset_summary',
            input=['quick_testset_summary_excel', 'aug_testset_summary'], output='final_testset_summary_excel',
            merged_keys=['testset'], method=choose_appropriate_summary
        ),
        *excel_parsing_blueprint(
            item_names={
                'xlsx': 'final_testset_summary_excel',
                'block': 'final_summary_block',
                'object': 'final_summary_object'
            }
        ),
    ]
    return quicksummary_transforms
