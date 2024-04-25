import itertools
import json
import math
import pandas as pd
from pandas.api.types import CategoricalDtype

from pysquared import Transform
import pysquared.transforms.transform_templates as templates

from .timings import METHOD_NAMES
from utils import confsearch


def fullreport_dataitems():
    return {
        'report_settings': {'type': 'object', 'keys': ['reporttype']},

        'sampling_rates_dfs': {'type': 'object', 'keys': ['reporttype', 'testset', 'testcase']},
        'sampling_rates_dfs_csv': {'type': 'file', 'mask': './finalreport/dataframes/sampling_rates/{reporttype}/{testset}_{testcase}.csv'},
        
        # 'experimental_energy_by_level': {'type': 'object', 'keys': ['testset', 'testcase', 'level']},
        'energy_distr_features_dfs': {'type': 'object', 'keys': ['reporttype', 'testset', 'testcase']},
        'energy_distr_features_dfs_csv': {'type': 'file', 'mask': './finalreport/dataframes/energy_distr_features/{reporttype}/{testset}_{testcase}.csv'},
        
        'expconformer_crmsd_dfs': {'type': 'object', 'keys': ['reporttype', 'testset', 'testcase']},
        'expconformer_crmsd_dfs_csv': {'type': 'file', 'mask': './finalreport/dataframes/expconformer_crmsd/{reporttype}/{testset}_{testcase}.csv'},

        'energy_distr_plots_finalreport': {'type': 'object', 'keys': ['reporttype', 'testset', 'testcase']},
        'diversity_plot_finalreport': {'type': 'object', 'keys': ['reporttype', 'testset', 'testcase']},

        'finalreport_pdf': {'type': 'file', 'mask': './finalreport/{reporttype}.pdf'},
    }

def sampling_rates_process(df: pd.DataFrame, **fixed_keys) -> pd.DataFrame:
    df['num_above_relenergy_0'] = 0
    df = df.melt(
        id_vars=[colname for colname in df.columns if not colname.startswith('num_above_relenergy_')],
        value_vars=[colname for colname in df.columns if colname.startswith('num_above_relenergy_')],
        var_name='relenergy_threshold', value_name='num_above_relenergy'
    )
    df['relenergy_threshold'] = df['relenergy_threshold'].str.extract(r'(\d+)').astype(int)
    unique_relenergy_thresholds = list(df['relenergy_threshold'].unique())
    relenergy_threshold_type = CategoricalDtype(categories=[
        *[
            i
            for i in sorted(unique_relenergy_thresholds)
            if i != 0
        ],
        0
    ], ordered=True)
    df['relenergy_threshold'] = df['relenergy_threshold'].astype(relenergy_threshold_type)
    df['relenergy_threshold'] = df['relenergy_threshold'].cat.rename_categories({
        i: f"E < {i} kcal/mol" if i != 0 else "No E threshold"
        for i in unique_relenergy_thresholds
    })

    relenergy_threshold_categories = df['relenergy_threshold'].cat.categories
    df['method'] = df['method'].replace(METHOD_NAMES)

    df['time_per_conf'] = df['time'] / (df['num_succ'] - df['num_above_relenergy'])
    df = (
        df.drop(
            ['plottype', 'testcase', 'nconf', 'timelimit', 'testset', 'num_above_relenergy', 'num_rmsdfail', 'num_succ', 'nunique', 'time'],
            axis=1
        )
        .pivot(index=['level', 'method'], columns='relenergy_threshold', values='time_per_conf')
        .reset_index()
        .rename_axis(None, axis=1)
        .sort_values(by=['level', *relenergy_threshold_categories])
        .reset_index(drop=True)
    )
    return df

def energy_distr_process(df: pd.DataFrame, expenergies, **fixed_keys) -> pd.DataFrame:
    
    df = (
        df
        .drop(['testset', 'testcase'], axis=1)
        .groupby(['method', 'level', 'esummary_type'])
        .agg(min_energy=('energy', 'min'))
        .reset_index()
    )

    levels = df['level'].unique()
    for level in levels:
        df.loc[len(df)] = {
            'method': 'experimental',
            'level': level,
            'esummary_type': 'experimental',
            'min_energy': expenergies[level],
        }
    min_energy_by_level = df.groupby('level')['min_energy'].transform('min')
    df['min_energy'] = df['min_energy'] - min_energy_by_level

    level_type = CategoricalDtype(categories=['mmff', 'gfnff'], ordered=True)
    df['level'] = df['level'].astype(level_type)

    df['method'] = df['method'].replace(METHOD_NAMES)
    df = (
        df[['esummary_type', 'level', 'method', 'min_energy']]
        .sort_values(by=['esummary_type', 'level', 'min_energy'])
        .reset_index(drop=True)
    )
    return df

def expconformer_crmsd_process(df: pd.DataFrame, **fixed_keys) -> pd.DataFrame:
    level_type = CategoricalDtype(categories=['mmff', 'gfnff'], ordered=True)

    df['method'] = df['method'].replace(METHOD_NAMES)
    df['level'] = df['level'].astype(level_type)
    result_df = (
        df[['plottype', 'level', 'method', 'min_crmsd']]
        .sort_values(by=['plottype', 'level', 'min_crmsd'])
        .reset_index(drop=True)
    )

    return result_df

def arrange_energy_distribution_plots(df: pd.DataFrame, **fixed_keys) -> dict[str, list[str]]:
    level_type = CategoricalDtype(categories=['mmff', 'gfnff'], ordered=True)
    df['level'] = df['level'].astype(level_type)
    style_type = CategoricalDtype(categories=['abstotal', 'abslocal', 'norm'], ordered=True)
    df['edist_style'] = df['edist_style'].astype(style_type)
    df = (
        df
        .sort_values(by=['level', 'esummary_type', 'edist_style'])
        .reset_index(drop=True)
    )

    result = {}
    for keys, group_df in df.groupby(['level', 'esummary_type'], observed=True):
        level, esummary_type = keys
        result[f"{esummary_type} ({level.upper()})"] = group_df.sort_values(by=['level', 'esummary_type', 'edist_style'])['path'].tolist()
    return result

def arrange_diversity_plots(df: pd.DataFrame, **fixed_keys) -> dict[str, list[str]]:
    level_type = CategoricalDtype(categories=['mmff', 'gfnff'], ordered=True)
    df['level'] = df['level'].astype(level_type)
    style_type = CategoricalDtype(categories=['basicOFive', 'vsrdkitOFiveNew', 'vsmtdOFive'], ordered=True)
    df['divplot_type'] = df['divplot_type'].astype(style_type)
    df = (
        df
        .sort_values(by=['level', 'divplot_type'])
        .reset_index(drop=True)
    )

    # raise Exception('AAAAAAAA')
    result = {}
    for keys, group_df in df.groupby(['level', 'divplot_type'], observed=True):
        level, divplot_type = keys
        result[f"{divplot_type.replace('OFive', '').replace('New', '')} ({level.upper()})"] = group_df['path'].tolist()
    return result


REPORT_SETTINGS = {
    'mainreport': {
        'sampling_rates': {
            'accept': lambda plottype, level, **kw: plottype == 'basic-doublelog',
            'process': sampling_rates_process,
        },
        'min_energies': {
            'accept_df': lambda testset, esummary_type, **kw: (
                # esummary_type in ('gfnpost','gfnonly','puremmff','vsrdkit') and
                esummary_type in ('basicmmff', 'vscrest-gfn', 'vscrest-gfn', 'vscrest-mmff', 'vsrdkit-mmff') and
                testset == 'macromodel'
            ),
            'filter_df': lambda df, **kw: df, # No filtering
            'process': energy_distr_process,
        },
        'expconformer_crmsd': {
            'accept': lambda plottype, method, level, **kw: (
                plottype in ('fastmethods-allmols', 'allmethods-heavymols', 'mcr-vs-rdkit', 'mcr-vs-mtd', 'mcr-vs-crest')
            ),
            'process': expconformer_crmsd_process,
        },
        'energy_distributions': {
            'accept': lambda esummary_type, **kw: (
                # esummary_type in ('puremmff',)
               esummary_type in ('basicmmff', 'vscrest-gfn', 'vsrdkit-mmff')
            ),
            'process': arrange_energy_distribution_plots,
        },
        'diversity_maps': {
            'accept': lambda divplot_type, **kw: divplot_type in ('basicOFive', 'vsrdkitOFiveNew', 'vsmtdOFive',),
            'process': arrange_diversity_plots,
        },
    },
}

# LOWEST EXPCONFORMER CRMSD
def load_expconformer_crmsd(dfs: pd.DataFrame, result_dfs, settings_item) -> None:
    for settings, settings_keys in settings_item:
        reporttype: str = settings_keys['reporttype']
        expconformer_crmsd_settings = settings['expconformer_crmsd']
        expconformer_crmsd_df = []
        for df, df_keys in dfs:
            for index, row in df.iterrows():
                conformer_data = {**df_keys, **row}
                if not expconformer_crmsd_settings['accept'](**conformer_data):
                    continue
                expconformer_crmsd_df.append(conformer_data)
        expconformer_crmsd_df = pd.DataFrame(expconformer_crmsd_df)

        testsets = expconformer_crmsd_df['testset'].unique()
        testcases = expconformer_crmsd_df['testcase'].unique()
        for testset, testcase in itertools.product(testsets, testcases):
            new_df: pd.DataFrame = (
                expconformer_crmsd_df[
                    (expconformer_crmsd_df['testset'] == testset) & 
                    (expconformer_crmsd_df['testcase'] == testcase)
                ]
                .reset_index(drop=True)
                .copy()
            )
            if len(new_df) == 0:
                continue

            new_df = expconformer_crmsd_settings['process'](
                new_df,
                reporttype=reporttype,
                testset=testset,
                testcase=testcase,
            )
            result_dfs.include_element(new_df, reporttype=reporttype, testset=testset, testcase=testcase)

# ENERGY DISTRIBUTION
def load_expenergies_by_level(input_json, output_energies):
    energies = {
        'testset': [],
        'testcase': [],
        'level': [],
        'energy': [],
    }
    def load_item(testset: str, testcase: str, level: str, energy: str):
        energies['testset'].append(testset)
        energies['testcase'].append(testcase)
        energies['level'].append(level)
        energies['energy'].append(energy)

    for json_path, json_keys in input_json:
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        testset = json_keys['testset']
        
        for testcase, case_data in json_data.items():
            if 'mmff' in case_data:
                load_item(testset=testset, testcase=testcase, level='mmff', energy=case_data['mmff'])
            if 'gfnff' in case_data or 'gfnffpost' in case_data:
                energy = min(
                    case_data[opttype]
                    for opttype in ('gfnff', 'gfnffpost')
                    if opttype in case_data
                )
                load_item(testset=testset, testcase=testcase, level='gfnff', energy=energy)
        
    ener_df = pd.DataFrame(energies)
    ener_df = (
        ener_df
        .groupby(['testset', 'testcase', 'level'])['energy']
        .min()
        .reset_index()
        .sort_values(by=['testcase'])
        .reset_index(drop=True)
    )

    for key, group in ener_df.groupby(['testset', 'testcase', 'level']):
        testset, testcase, level = key
        energies = group['energy'].tolist()
        assert len(energies) == 1
        output_energies.include_element(
            energies[0],
            testset=testset,
            testcase=testcase,
            level=level,
        )

def load_energy_distr_features(dfs: pd.DataFrame, result_dfs, expenergies, settings_item) -> None:
    for settings, settings_keys in settings_item:
        reporttype: str = settings_keys['reporttype']
        energy_distr_settings = settings['min_energies']
        energy_distr_dfs: list[pd.DataFrame] = []
        for df, df_keys in dfs:
            if not energy_distr_settings['accept_df'](**df_keys):
                continue
            changed_df = energy_distr_settings['filter_df'](df, **df_keys).copy()
            changed_df['esummary_type'] = df_keys['esummary_type']
            changed_df['testset'] = df_keys['testset']
            changed_df['level'] = df_keys['level']
            energy_distr_dfs.append(changed_df)
        energy_distr_df = pd.concat([df for df in energy_distr_dfs], ignore_index=True)
        ic(energy_distr_df)

        testsets = energy_distr_df['testset'].unique()
        testcases = energy_distr_df['testcase'].unique()
        for testset, testcase in itertools.product(testsets, testcases):
            new_df: pd.DataFrame = (
                energy_distr_df[
                    (energy_distr_df['testset'] == testset) & 
                    (energy_distr_df['testcase'] == testcase)
                ]
                .reset_index(drop=True)
                .copy()
            )
            if len(new_df) == 0:
                continue
        
            cur_expenergies = {
                keys['level']: expenergy
                for expenergy, keys in expenergies
                if keys['testcase'] == testcase and keys['testset'] == testset
            }

            new_df = energy_distr_settings['process'](
                new_df,
                expenergies=cur_expenergies,
                reporttype=reporttype,
                testset=testset,
                testcase=testcase,
            )
            result_dfs.include_element(new_df, reporttype=reporttype, testset=testset, testcase=testcase)


# SAMPLING RATES
def load_sampling_rates(dfs: pd.DataFrame, result_dfs, settings_item) -> None:
    for settings, settings_keys in settings_item:
        reporttype: str = settings_keys['reporttype']
        sampling_rates_settings = settings['sampling_rates']
        sampling_rates_df = []
        for df, df_keys in dfs:
            for index, row in df.iterrows():
                timing_data = {**df_keys, **row}
                if not sampling_rates_settings['accept'](**timing_data):
                    continue
                sampling_rates_df.append(timing_data)
        sampling_rates_df = pd.DataFrame(sampling_rates_df)

        ic(sampling_rates_df)

        testsets = sampling_rates_df['testset'].unique()
        testcases = sampling_rates_df['testcase'].unique()
        for testset, testcase in itertools.product(testsets, testcases):
            new_df: pd.DataFrame = (
                sampling_rates_df[
                    (sampling_rates_df['testset'] == testset) & 
                    (sampling_rates_df['testcase'] == testcase)
                ]
                .reset_index(drop=True)
                .copy()
            )
            if len(new_df) == 0:
                continue

            new_df = sampling_rates_settings['process'](
                new_df,
                reporttype=reporttype,
                testset=testset,
                testcase=testcase,

            )
            result_dfs.include_element(new_df, reporttype=reporttype, testset=testset, testcase=testcase)


def prepare_diversity_plots(plots, result_paths, settings_item):
    for settings, settings_keys in settings_item:
        reporttype: str = settings_keys['reporttype']
        diversity_settings = settings['diversity_maps']
        diversity_df = []
        for plot_path, plot_keys in plots:
            if not diversity_settings['accept'](**plot_keys):
                continue
            diversity_df.append({**plot_keys, 'path': plot_path})
        diversity_df = pd.DataFrame(diversity_df)
        ic(diversity_df)

        testsets = diversity_df['testset'].unique()
        testcases = diversity_df['testcase'].unique()
        for testset, testcase in itertools.product(testsets, testcases):
            new_df: pd.DataFrame = (
                diversity_df[
                    (diversity_df['testset'] == testset) & 
                    (diversity_df['testcase'] == testcase)
                ]
                .reset_index(drop=True)
                .copy()
            )
            if len(new_df) == 0:
                continue

            # ic('use', testcase, new_df)
            new_df = diversity_settings['process'](
                new_df,
                reporttype=reporttype,
                testset=testset,
                testcase=testcase,

            )
            ic('use', testcase, new_df)
            result_paths.include_element(new_df, reporttype=reporttype, testset=testset, testcase=testcase)


def prepare_energy_distr_plots(plots, result_paths, settings_item):
    for settings, settings_keys in settings_item:
        reporttype: str = settings_keys['reporttype']
        energy_distributions_settings = settings['energy_distributions']
        energy_distributions_df = []
        for plot_path, plot_keys in plots:
            if not energy_distributions_settings['accept'](**plot_keys):
                continue
            energy_distributions_df.append({**plot_keys, 'path': plot_path})
        energy_distributions_df = pd.DataFrame(energy_distributions_df)

        ic(energy_distributions_df)

        testsets = energy_distributions_df['testset'].unique()
        testcases = energy_distributions_df['testcase'].unique()
        for testset, testcase in itertools.product(testsets, testcases):
            new_df: pd.DataFrame = (
                energy_distributions_df[
                    (energy_distributions_df['testset'] == testset) & 
                    (energy_distributions_df['testcase'] == testcase)
                ]
                .reset_index(drop=True)
                .copy()
            )
            if len(new_df) == 0:
                continue

            new_df = energy_distributions_settings['process'](
                new_df,
                reporttype=reporttype,
                testset=testset,
                testcase=testcase,

            )
            result_paths.include_element(new_df, reporttype=reporttype, testset=testset, testcase=testcase)


def build_finalreport_pdf(text_items: dict[str, any], image_items: dict[str, any], result_pdf):
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from chemscripts.imageutils import HImageLayout, resize_to_width
    from io import BytesIO
    import PIL

    all_testcases = set(
        (keys['testset'], keys['testcase'])
        for _, keys in itertools.chain(*text_items.values(), *image_items.values())
    )
    all_testcases = sorted(list(all_testcases))
    heavy_testcases = ['pdb2IYA', 'csdMIWTER', 'csdYIVNOG', 'pdb2C6H', 'pdb3M6G', 'pdb2QZK', 'pdb1NWX', 'csdRECRAT', 'csdFINWEE10', 'csdRULSUN']
    heavy_testcases = [('macromodel', x) for x in heavy_testcases]
    easy_testcases = [
        x
        for x in all_testcases
        if x not in heavy_testcases
    ]

    text_items = {
        item_name: {
            (keys['testset'], keys['testcase']): obj
            for obj, keys in item
        }
        for item_name, item in text_items.items()
    }

    image_items = {
        item_name: {
            (keys['testset'], keys['testcase']): obj
            for obj, keys in item
        }
        for item_name, item in image_items.items()
    }

    def format_value(v):
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                return '—'
            return str(round(v, 4))
        else:
            return Paragraph(v)

    def df2table(df):
        return Table(
            [[Paragraph(col) for col in df.columns]] + 
            [
                [
                    format_value(v)
                    for v in row
                ]
                for row in df.values.tolist()
            ], 
            style=[
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('LINEBELOW',(0,0), (-1,0), 1, colors.black),
                ('INNERGRID', (0,0), (-1,-1), 0.25, colors.black),
                ('BOX', (0,0), (-1,-1), 1, colors.black),
                ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.lightgrey, colors.white]),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ],
            hAlign='CENTER'
        )

    def images_as_row(image_paths: list[str]):
        main_image = HImageLayout()
        for path in image_paths:
            main_image.insert(PIL.Image.open(path), type='middle')
        main_image = main_image.build()

        image_stream = BytesIO()
        main_image.save(image_stream, 'PNG')
        image_stream.seek(0)
        main_width, main_height = main_image.size
        return Image(image_stream, width=letter[0], height=int(letter[0]/main_width*main_height))

    styles = getSampleStyleSheet()
    INTRO_MESSAGE = """\
This document contains detailed benchmark summaries grouped for each of 150 test molecules.
Record of each molecule includes several sections (some can be omitted):
* "Sampling speed per conformer" provides sampling rates for each method and 3 energy windows + no energy cutoffs. This is the data used to plot Figure 3.

* "Lowest energies of sampled conformers" provides energy in kcal/mol of the lowest generated conformer for each method.
    Logical subsections:
    1. "basicmmff" groups the core sampling runs of the benchmark with MCR + 2 or 5 reference method
    2. "experimental" relative energies of experimental conformers
    3. "vscrest-*" - detailed comparison of MCR with CREST. Includes "MCR (in CREST time)" - the MCR run for time interval required by CREST.
    4. "vsrdkit-mmff" - detailed comparison of MCR with RDKit. Includes "ETKDGv1" from older version of RDKit 2022.03.5, and "MCR (N of ETKDGv3 2024)" - the MCR run to generate the number of conformers produced by ETKDGv3.

* "Lowest CRMSD with experimental conformer" provides smaller CRMSD value in Angstroms w.r.t. experimental conformer for each method.
    Logical subsections:
    1. "allmethods-heavymols" groups the core sampling runs of the benchmark with MCR + 5 reference method. Only for 10 test molecules.
    2. "fastmethods-allmols" groups the core sampling runs of the benchmark with MCR + 2 fast reference method (ETKDGv3 and MTD). Included for all 150 test molecules.
    3. "mcr-vs-crest" - detailed comparison of MCR with CREST. Includes "MCR (in CREST time)" - the MCR run for time interval required by CREST.
    4. "mcr-vs-rdkit" - detailed comparison of MCR with RDKit. Includes "ETKDGv1" from older version of RDKit 2022.03.5, and "MCR (N of ETKDGv3 2024)" - the MCR run to generate the number of conformers produced by ETKDGv3.
    5. "mcr-vs-mtd" - detailed comparison of MCR with MTD. Includes "MCR (N of XTB MTD)" - the MCR run to generate the number of conformers produced by MTD.

* "Energy distribution plots" illustrate energy densities for each method. Order of plots: non-normalized energy densities - close-up at small energy segment of the non-normalized plot - normalized energy densities. Vertical dashed line highlights the energy of the experimentally observed conformer optimized at the same level of theory.
    Logical subsections:
    1. "basicmmff" groups the core sampling runs of the benchmark with MCR + 2 or 5 reference method
    2. "vscrest-gfn" - detailed comparison of MCR with CREST. Includes "MCR (in CREST time)" - the MCR run for time interval required by CREST.
    3. "vsrdkit-mmff" - detailed comparison of MCR with RDKit. Includes "ETKDGv1" from older version of RDKit 2022.03.5.

* "Diversity plots" illustrate conformational space coverage by each method.
    Logical subsections:
    1. "basic" groups the core sampling runs of the benchmark with MCR + 2 or 5 reference method. These were used for Figure 4.
    2. "vsrdkit" - detailed comparison of MCR with RDKit. Includes "ETKDGv1" from older version of RDKit 2022.03.5. Plotted with reference to "MCR (N of ETKDGv3)" - the MCR run to generate the number of conformers produced by ETKDGv3.
    3. "vsmtd" - detailed comparison of MCR with MTD. Plotted with reference to "MCR (N of XTB MTD)" - the MCR run to generate the number of conformers produced by XTB MTD.
"""
    elements = [Paragraph(INTRO_MESSAGE.replace('\n', '<br></br>'))]#, styles['Heading3'])]
    for testset, testcase in heavy_testcases + easy_testcases:
        elements.append(Paragraph(confsearch.format_testcase(testcase), styles['Heading1']))
        for item_name, item_elements in text_items.items():
            if (testset, testcase) not in item_elements:
                continue
            if (testset, testcase) not in item_elements:
                continue
            elements.append(Paragraph(f"{item_name}:", styles['Heading4']))
            elements.append(df2table(item_elements[testset, testcase]))
            elements.append(Spacer(1, 12))
        
        for image_header, subimages in image_items.items():
            if (testset, testcase) not in subimages:
                continue
            elements.append(Paragraph(f"{image_header}:", styles['Heading4']))
            for image_line_header, image_line in subimages[testset, testcase].items():
                elements.append(Paragraph(f"{image_line_header}:", styles['Heading6']))
                elements.append(images_as_row(image_line))
            elements.append(Spacer(1, 12))

    pdf_path: str = result_pdf.get_path()
    doc = SimpleDocTemplate(pdf_path, pagesize=(letter[0] + 60, letter[1]), leftMargin=0, rightMargin=0, topMargin=0, bottomMargin=0, title="SI Detailed benchmark summaries")
    doc.build(elements)
    result_pdf.include_element(pdf_path)


def fullreport_transforms(ds, main_logger) -> list[Transform]:
    for reporttype, report_settings in REPORT_SETTINGS.items():
        ds.report_settings.include_element(report_settings, reporttype=reporttype)

    fullreport_transforms: list[Transform] = [
        # Load sampling rates
        templates.exec('load_sampling_rates',
            input=['timing_df', 'report_settings'], output='sampling_rates_dfs',
            merged_keys=['relativity', 'level', 'plottype', 'reporttype'],
            method=lambda timing_df, sampling_rates_dfs, report_settings, **kw:
                load_sampling_rates(dfs=timing_df, result_dfs=sampling_rates_dfs, settings_item=report_settings)
        ),
        templates.pd_to_csv('dump_sampling_rates_dfs',
            input='sampling_rates_dfs', output='sampling_rates_dfs_csv', sep=';', index=False
        ),
        templates.pd_from_csv('load_sampling_rates_dfs',
            input='sampling_rates_dfs_csv', output='sampling_rates_dfs', sep=';'
        ),
        
        # Load energy distributions
        # templates.exec('load_expenergies_by_level',
        #     input='experimental_energies_json', output='experimental_energy_by_level',
        #     merged_keys=['esummary_type', 'testset'],
        #     method=lambda experimental_energies_json, experimental_energy_by_level:
        #         load_expenergies_by_level(input_json=experimental_energies_json, output_energies=experimental_energy_by_level)
        # ),
        templates.exec('load_energy_distribution_features',
            input=['energy_distribution_summary', 'expconformer_relenergies', 'report_settings'], output='energy_distr_features_dfs',
            merged_keys=['relativity', 'level', 'esummary_type', 'testset', 'testcase', 'reporttype'],
            method=lambda energy_distribution_summary, energy_distr_features_dfs, expconformer_relenergies, report_settings, **kw:
                load_energy_distr_features(
                    dfs=energy_distribution_summary,
                    result_dfs=energy_distr_features_dfs,
                    expenergies=expconformer_relenergies,
                    settings_item=report_settings
                )
        ),
        templates.pd_to_csv('dump_energy_distr_features',
            input='energy_distr_features_dfs', output='energy_distr_features_dfs_csv', sep=';', index=False
        ),
        templates.pd_from_csv('load_energy_distr_features',
            input='energy_distr_features_dfs_csv', output='energy_distr_features_dfs', sep=';'
        ),

        # To show the lowest CRMSD with experimental conformer for each method
        templates.pd_from_csv('load_expconformer_crmsd_df',
            input='compare_df_csv', output='compare_df', sep=';'
        ),
        templates.exec('get_lowest_expconformer_crmsd',
            input=['compare_df', 'report_settings'], output='expconformer_crmsd_dfs',
            merged_keys=['plottype', 'reporttype'],
            method=lambda compare_df, expconformer_crmsd_dfs, report_settings, **kw:
                load_expconformer_crmsd(
                    dfs=compare_df,
                    result_dfs=expconformer_crmsd_dfs,
                    settings_item=report_settings
                )
        ),
        templates.pd_to_csv('dump_processed_expconformer_crmsd_df',
            input='expconformer_crmsd_dfs', output='expconformer_crmsd_dfs_csv', sep=';', index=False
        ),
        templates.pd_from_csv('load_processed_expconformer_crmsd_df',
            input='expconformer_crmsd_dfs_csv', output='expconformer_crmsd_dfs', sep=';'
        ),

        # Plot energy distributions
        templates.exec('prepare_energy_distr_plots_paths',
            input=['energy_distr_single_plot', 'report_settings'], output='energy_distr_plots_finalreport',
            merged_keys=['relativity', 'level', 'testset', 'testcase', 'esummary_type', 'edist_style', 'reporttype',],
            method=lambda energy_distr_single_plot, energy_distr_plots_finalreport, report_settings, **kw:
                prepare_energy_distr_plots(
                    plots=energy_distr_single_plot,
                    result_paths=energy_distr_plots_finalreport,
                    settings_item=report_settings
                )
        ),

        # Load diversity plots
        templates.exec('prepare_diversity_plot_plots_paths',
            input=['diversity_plot_png', 'report_settings'], output='diversity_plot_finalreport',
            merged_keys=['relativity', 'level', 'testset', 'testcase', 'divplot_type', 'reporttype',],
            method=lambda diversity_plot_png, diversity_plot_finalreport, report_settings, **kw:
                prepare_diversity_plots(
                    plots=diversity_plot_png,
                    result_paths=diversity_plot_finalreport,
                    settings_item=report_settings
                )
        ),

        templates.exec('build_finalreport_pdf',
            input=['sampling_rates_dfs', 'energy_distr_features_dfs', 'expconformer_crmsd_dfs', 'energy_distr_plots_finalreport', 'diversity_plot_finalreport'],
            output='finalreport_pdf', merged_keys=['testset', 'testcase'],
            method=lambda sampling_rates_dfs, energy_distr_features_dfs, expconformer_crmsd_dfs, energy_distr_plots_finalreport, diversity_plot_finalreport, finalreport_pdf:
                build_finalreport_pdf(
                    text_items={
                        'Sampling speed per conformer': sampling_rates_dfs,
                        'Lowest energies of sampled conformers': energy_distr_features_dfs,
                        'Lowest CRMSD with experimental conformer': expconformer_crmsd_dfs,
                    },
                    image_items={
                        'Energy distribution plots': energy_distr_plots_finalreport,
                        'Diversity plots': diversity_plot_finalreport,
                    },
                    result_pdf=finalreport_pdf,
                )
        ),
    ]
    return fullreport_transforms
