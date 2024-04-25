import json
from typing import Tuple, Iterator

import pandas as pd
from pandas.api.types import CategoricalDtype

from pysquared import Transform
from pysquared import TransformStateFactories as ret
import pysquared.transforms.transform_templates as templates

from .timings import METHOD_NAMES, METHOD_COLORS
from utils import confsearch
from chemscripts.utils import H2KC

#
# Plot density of E-distribution for CREST and MCR on two levels of theory: MMFF & GFNFF
#


def energy_distribution_dataitems() -> dict[str, dict]:
    return {
        # Build E-distr summary
        'energy_distribution_summary': {'type': 'object', 'keys': ['relativity', 'level', 'testset', 'esummary_type']},
        'energy_distribution_csv': {'type': 'file', 'mask': './energy_distribution/dataframes/ener_distrs/{relativity}_{level}/{testset}_{esummary_type}.csv'},
        'energy_distribution_single_df': {'type': 'object', 'keys': ['relativity', 'level', 'testset', 'testcase', 'esummary_type']},
        
        'placeholder_none_expenergy': {'type': 'object', 'keys': ['relativity', 'level', 'testset', 'testcase', 'esummary_type']},
        'edistr_expconformer_relenergies': {'type': 'object', 'keys': ['relativity', 'level', 'testset', 'testcase', 'esummary_type']},
        'experimental_energy_restricted': {'type': 'object', 'keys': ['relativity', 'level', 'testset', 'testcase', 'esummary_type']},
        'energy_distr_plotting_data': {'type': 'object', 'keys': ['relativity', 'level', 'testset', 'testcase', 'esummary_type']},
        'energy_distr_single_plot': {'type': 'file', 'mask': './energy_distribution/single_plots/{relativity}_{level}/{testset}_{esummary_type}/{testcase}_{edist_style}.png'},
        'esummary_typed_report_pdf': {'type': 'file', 'mask': './energy_distribution/reports/{testset}_{esummary_type}.pdf'},
    }

# Since this is a density plot, here we only care about showing as many datapoint as we can
MCR_VS_CREST_METHODS = ('ringo-vs-crest', 'crestOld')
MCR_VS_MTD_METHODS = ('mtd', 'ringo')
MCR_VS_RDKIT_METHODS = ('ringo', 'ETKDGv3-2024', 'ETKDGv1') # 'ringo-vs-rdkit',

BASIC_METHODS = ('ringo', 'ETKDGv3', 'mtd', 'crestOld', 'mmringOld', 'mmbasicOld')
MCR_VS_CREST_MOLECULES = ('pdb2IYA', 'csdMIWTER', 'csdRULSUN', 'csdYIVNOG', 'pdb2C6H', 'pdb3M6G', 'pdb1NWX', 'csdRECRAT', 'csdFINWEE10', 'pdb2QZK')

ENERGY_DISTR_TYPES = {
    'basicmmff': {
        'accept': lambda relativity, **kw: relativity == 'basic',
    },
    'vscrest-gfn': {
        'accept': lambda relativity, level, method, **kw:
            (relativity == 'global' and level == 'gfnff' and method in MCR_VS_CREST_METHODS),
    },
    'vscrest-mmff': {
        'accept': lambda relativity, level, method, **kw:
            (relativity == 'global' and level == 'mmff' and method in (*MCR_VS_CREST_METHODS, 'ringo')),
    },
    'vsrdkit-mmff': {
        'accept': lambda relativity, level, method, **kw:
            (relativity == 'global' and level == 'mmff' and method in (*MCR_VS_RDKIT_METHODS, 'ringo-vs-rdkit2024')),
    },
    'vsmtd-mmff': {
        'accept': lambda relativity, level, method, **kw:
            (relativity == 'global' and level == 'mmff' and method in (*MCR_VS_MTD_METHODS, 'ringo-vs-mtd')),
    },
}


def merge_energy_info(energies_json, result_df) -> None:
    for esummary_type, esummary_settings in ENERGY_DISTR_TYPES.items():
        keys_for_separate_dfs = {
            (keys['relativity'], keys['level'], keys['testset'])
            for path, keys in energies_json    
            if esummary_settings['accept'](**keys)
        }
        for relativity, level, testset in keys_for_separate_dfs:
            df = {
                'method': [],
                'testcase': [],
                'energy': [],
            }
            for energy_json_path, keys in energies_json:
                if not (esummary_settings['accept'](**keys) and
                    keys['relativity'] == relativity and
                    keys['level'] == level and
                    keys['testset'] == testset
                ):
                    continue
                
                with open(energy_json_path, 'r') as f:
                    relenergies: list[float] = json.load(f)['relative']

                for energy in relenergies:
                    df['method'].append(keys['method'])
                    df['testcase'].append(keys['testcase'])
                    df['energy'].append(energy)
            df = pd.DataFrame(df)
            result_df.include_element(
                df,
                relativity=relativity,
                level=level,
                esummary_type=esummary_type,
                testset=testset
            )

def unpack_energy_distributions_summary(df: pd.DataFrame) -> Iterator[tuple[pd.DataFrame, dict[str, str]]]:
    return (
        (
            df[df['testcase'] == testcase],
            {'testcase': testcase}
        )
        for testcase in df['testcase'].unique()
    )


def select_relative_expenergies(energies, dfs, target_item) -> None:
    for df, keys in dfs:
        relativity, level, testset, testcase, esummary_type = (
            keys['relativity'], keys['level'], keys['testset'], keys['testcase'], keys['esummary_type']
        )
        
        energy_keys = {'relativity': relativity, 'level': level, 'testset': testset, 'testcase': testcase}
        if energies.contains_keys(keys=energy_keys):
            relenergy = energies.access_element(**energy_keys)
        else:
            relenergy = None
        
        target_item.include_element(relenergy, **keys)


def energy_distribution_transforms(ds, main_logger, execscripts_dir: str, maxproc=1) -> list[Transform]:
    METHOD_ORDERINGS = {
        'gfnonly': [
            'ringo-vs-crest',
            # 'ringoCN-vs-crest',
            'crestOld',
        ],
        # 'gfnpost': [
        #     'ringo-vs-crest',
        #     'ringoCN-vs-crest',
        #     'crestOld',
        # ],
        'basicmmff': [
            'ringo',
            'ETKDGv3-2024',
            'mtd',
            'crestOld',
            'mmringOld',
            'mmbasicOld',
        ],
        'vsrdkit-mmff': [
            'ringo',
            # 'ringo-vs-rdkit',
            # 'ringoCN-vs-rdkit',
            'ETKDGv3-2024',
            'ETKDGv1',
        ],
        'vscrest-mmff': [
            'ringo',
            'ringo-vs-crest',
            # 'ringoCN-vs-crest',
            'crestOld',
        ],
        'vscrest-gfn': [
            # 'ringo',
            'ringo-vs-crest',
            # 'ringoCN-vs-crest',
            'crestOld',
        ],
        'vsmtd-mmff': [
            'ringo',
            'mtd',
            # 'ringo-vs-mtd',
        ],
    }

    def build_plotting_data(
        df: pd.DataFrame,
        esummary_type: str,
    ) -> dict[str, pd.DataFrame | float]:
        res_df = df.copy()
        
        method_ordering = METHOD_ORDERINGS[esummary_type]
        res_df = res_df[res_df['method'].isin(method_ordering)]
        res_df['method'] = res_df['method'].replace(METHOD_NAMES)

        method_ordering_renamed = [METHOD_NAMES[x] for x in method_ordering]
        method_type = CategoricalDtype(categories=method_ordering_renamed, ordered=True)
        res_df['method'] = res_df['method'].astype(method_type)

        return res_df


    def plot_energy_distrbution(
        df: pd.DataFrame,
        expenergy: float,
        png_item,
        esummary_type: str,
        level: str,
    ) -> Iterator[Tuple[str, dict[str, str]]]:
        from plotnine.geoms.geom import geom
        from plotnine import ggplot, aes, after_stat, geom_density, geom_vline, scale_color_manual, xlim, geom_point, theme_bw, theme, element_blank, element_line, coord_cartesian, element_text, element_rect, labs, scale_y_continuous, scale_fill_manual

        normal_theme = (theme_bw() +
            theme(panel_grid_major = element_blank(),
                panel_grid_minor = element_blank(),
                panel_border = element_rect(colour='black', fill=None, size=1),
                axis_line = element_line(colour='black'),
                axis_title = element_text(size=16, face='bold', ma='center'),
                axis_text = element_text(size=14),
                legend_title = element_text(size=14, face='bold'),
                legend_text = element_text(size=14),
                figure_size=(6, 4)
            )
        )

        ADJUST_FACTOR = 0.3
        add_limits = lambda plot, a, b: plot + xlim(a, b + 5.0) + coord_cartesian(xlim=(a, b))
        def abslocal_tweak(plot):
            if expenergy is not None:
                threshold = expenergy
            else:
                threshold = 0.0
            return add_limits(plot, 0, threshold + 10.0) + geom_density(aes(y=after_stat("count")), alpha=0.3, adjust=ADJUST_FACTOR)
        EDISTR_TWEAKS = {
            'norm': lambda plot: add_limits(plot, 0.0, 50.0) + geom_density(alpha=0.3, adjust=ADJUST_FACTOR),
            'abstotal': lambda plot: add_limits(plot, 0, 50.0) + geom_density(aes(y=after_stat("count")), alpha=0.3, adjust=ADJUST_FACTOR),
            'abslocal': abslocal_tweak,
        }

        png_names = {}
        for edist_style, modify_plot in EDISTR_TWEAKS.items():
            plot = (
                ggplot(df, aes(x='energy', fill='method')) +
                labs(x=f'{level.upper()} energy, kcal/mol', y='Energy density')
            )
            plot = (
                modify_plot(plot) +
                scale_y_continuous(breaks=None) +
                scale_fill_manual(values=METHOD_COLORS) +
                normal_theme
                # conditional(edist_style != 'abslocal', theme(legend_position='none')) +
                # conditional(edist_style == 'mmff', theme(axis_title_x=element_blank()))
            )
            if expenergy is not None:
                plot += geom_vline(xintercept=expenergy, linetype='dashed')

            png_names[edist_style] = png_item.get_path(edist_style=edist_style)
            plot.save(png_names[edist_style], verbose=False)
        
        return (
            (png_name, {'edist_style': edist_style})
            for edist_style, png_name in png_names.items()
        )


    def build_esummary_typed_reports(
        plots, # Input
        pdf_item, # Output
    ) -> None:
        from chemscripts.imageutils import HImageLayout, VImageLayout, pil_to_reportlab
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from PIL import Image

        # Columns
        STYLE_ORDER = ['norm', 'abstotal', 'abslocal']
        # Rows
        LEVEL_ORDER = ['mmff', 'gfnff']
        TITLE_SIZE = 48

        pdf_path: str = pdf_item.get_path()
        c = canvas.Canvas(pdf_path, pagesize=letter)
        all_testcases = set(
            keys['testcase']
            for _, keys in plots
        )
        all_testcases = sorted(list(all_testcases))

        for testcase in all_testcases:
            current_pngs = {
                (keys['level'], keys['edist_style']): Image.open(png_name)
                for png_name, keys in plots
                if keys['testcase'] == testcase
            }
            cur_levels = set(level for level, edist_style in current_pngs.keys())

            plot_rows = {
                level: HImageLayout()
                for level in LEVEL_ORDER
                if level in cur_levels
            }
            for level, row_layout in plot_rows.items():
                for style in STYLE_ORDER:
                    row_layout.insert(current_pngs[level, style], type='middle')
                
            main_image = VImageLayout()
            for level in LEVEL_ORDER:
                if level in cur_levels:
                    main_image.insert(plot_rows[level].build(), type='middle')
            
            main_image = main_image.build()
            c.setPageSize((main_image.size[0], main_image.size[1] + TITLE_SIZE))
            c.drawImage(pil_to_reportlab(main_image), 0, 0, width=main_image.width, height=main_image.height)
            c.setFont("Helvetica", 32)
            c.drawString(
                0, main_image.size[1],
                f"{confsearch.format_testcase(testcase)}. Left - normalized. Middle, right - non-normalized. Dashed - experimental conformer energy"
            )
            c.showPage()
        
        c.save()
        pdf_item.include_element(pdf_path)


    energy_distribution_transforms: list[Transform] = [
        # Load energies and compile dataframe
        templates.exec('build_energy_distribution_summary',
            input='ensemble_relenergies_json', output='energy_distribution_summary',
            merged_keys=['relativity', 'level', 'testset', 'testcase', 'method', 'timelimit'],
            method=lambda ensemble_relenergies_json, energy_distribution_summary:
                merge_energy_info(
                    energies_json=ensemble_relenergies_json,
                    result_df=energy_distribution_summary,
                )
        ),
        templates.pd_to_csv('energy_distribution_to_csv',
            input='energy_distribution_summary', output='energy_distribution_csv', sep=';', index=False
        ),
        templates.pd_from_csv('energy_distribution_from_csv',
            input='energy_distribution_csv', output='energy_distribution_summary', sep=';'
        ),
        templates.map('unpack_energy_distributions_summary',
            input='energy_distribution_summary', output='energy_distribution_single_df',
            mapping=lambda energy_distribution_summary, **kw:
                unpack_energy_distributions_summary(df=energy_distribution_summary),
        ),

        # Organize data for plotting figures
        templates.exec('select_relative_expenergies_for_edistr',
            input=['expconformer_relenergies', 'energy_distribution_single_df'],
            output='edistr_expconformer_relenergies',
            merged_keys=['relativity', 'level', 'testset', 'testcase', 'esummary_type'],
            method=lambda expconformer_relenergies, energy_distribution_single_df, edistr_expconformer_relenergies, **kw:
                select_relative_expenergies(
                    dfs=energy_distribution_single_df,
                    energies=expconformer_relenergies,
                    target_item=edistr_expconformer_relenergies
                )
        ),

        templates.map('build_plotting_data',
            input='energy_distribution_single_df',
            output='energy_distr_plotting_data', aware_keys=['esummary_type'],
            mapping=lambda energy_distribution_single_df, **kw:
                build_plotting_data(df=energy_distribution_single_df, **kw)
        ),
        templates.map('create_E_distribution_plots',
            input=['energy_distr_plotting_data', 'edistr_expconformer_relenergies'], output='energy_distr_single_plot',
            aware_keys=['level', 'esummary_type'], # 'level' is for Y-axis label
            mapping=lambda energy_distr_plotting_data, edistr_expconformer_relenergies, energy_distr_single_plot, **keys:
                plot_energy_distrbution(
                    df=energy_distr_plotting_data,
                    expenergy=edistr_expconformer_relenergies,
                    png_item=energy_distr_single_plot,
                    **keys
                )
        ),
        templates.exec('build_esummary_typed_reports',
            input='energy_distr_single_plot', output='esummary_typed_report_pdf',
            merged_keys=['relativity', 'level', 'edist_style', 'testcase'],
            method=lambda energy_distr_single_plot, esummary_typed_report_pdf:
                build_esummary_typed_reports(plots=energy_distr_single_plot, pdf_item=esummary_typed_report_pdf)
        ),
    ]
    return energy_distribution_transforms
