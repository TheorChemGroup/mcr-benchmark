import fnmatch
import pandas as pd
from pandas.api.types import CategoricalDtype

from pysquared import Transform
import pysquared.transforms.transform_templates as templates

METHOD_NAMES = {
    'ringo': 'MCR',
    'ETKDGv3-2024': 'ETKDGv3 2024',
    'mtd': 'XTB MTD',
    'mmbasicOld': 'MacroModel MCMM',
    'mmringOld': 'MacroModel MCS',
    'crestOld': 'CREST',
    
    # Rdkit older versions
    'ETKDGv1': 'ETKDGv1',
    # 'ETKDGv3': 'ETKDGv3',
    # 'rdkitOld': 'ETKDG old',
}
METHOD_NAMES = {
    **METHOD_NAMES,
    # Same number of conformers
    'ringo-vs-rdkit2024': f"{METHOD_NAMES['ringo']} (N of {METHOD_NAMES['ETKDGv3-2024']})",
    'ringo-vs-mtd': f"{METHOD_NAMES['ringo']} (N of {METHOD_NAMES['mtd']})",
    # Same execution time
    'ringo-vs-crest': f"{METHOD_NAMES['ringo']} (in {METHOD_NAMES['crestOld']} time)",
    'ringo-vs-mmring': f"{METHOD_NAMES['ringo']} (in {METHOD_NAMES['mmringOld']} time)",
}

METHOD_COLORS = {
    # Basic methods
    'ringo': '#e41a1c', # red
    'ETKDGv3-2024': '#377eb8', # blue
    'mtd': '#4daf4a', # green
    'mmbasicOld': '#984ea3', # violet
    'mmringOld': '#a65628', # brown
    'crestOld': '#f781bf', # pink

    # Rdkit older versions
    'ETKDGv1': '#02818a', # strong blue
    # 'ETKDGv3': '#000000', # black
    # 'rdkitOld': '#7bccc4', # cyanish blue

    # MCR comparisons
    'ringo-vs-rdkit2024': '#a6cee3', # weak blue
    'ringo-vs-mtd': '#b2df8a', # weak green
    'ringo-vs-crest': '#cab2d6', # weak pink
    'ringo-vs-mmring': None,#'#cab2d6', # weak pink
}
METHOD_COLORS = {
    METHOD_NAMES[method_raw_name]: color
    for method_raw_name, color in METHOD_COLORS.items()
}

METHOD_ORDERING = [
    # Basic methods
    'ringo',
    'mtd',
    'ETKDGv3-2024',
    'crestOld',
    'mmringOld',
    'mmbasicOld',
    
    # Rdkit older versions
    'ETKDGv1',
    # 'ETKDGv3',
    # 'rdkitOld',
    
    # MCR comparisons
    'ringo-vs-rdkit2024',
    'ringo-vs-mtd',
    'ringo-vs-crest',
    'ringo-vs-mmring',
]
METHOD_ORDERING = [METHOD_NAMES[x] for x in METHOD_ORDERING]

HEAVY_TESTCASES = ['pdb2IYA', 'csdMIWTER', 'csdYIVNOG', 'pdb2C6H', 'pdb3M6G', 'pdb2QZK', 'pdb1NWX', 'csdRECRAT', 'csdFINWEE10', 'csdRULSUN']


def timings_dataitems():
    return {
        # Timing plots
        'timing_df': {'type': 'object', 'keys': ['relativity', 'level', 'plottype']},
        'timings_raw_csv': {'type': 'file', 'mask': './timings_analysis/dataframes/{relativity}_{level}_{plottype}_dfraw.csv'},

        'timing_df_with_data': {'type': 'object', 'keys': ['relativity', 'level', 'plottype']},
        'timing_df_processed': {'type': 'object', 'keys': ['relativity', 'level', 'plottype']},
        'timing_plot_settings': {'type': 'object', 'keys': ['plottype']},
        'timings_final_csv': {'type': 'file', 'mask': './timings_analysis/dataframes/{relativity}_{level}_{plottype}_df.csv'},
        'timings_plot_svg': {'type': 'file', 'mask': './timings_analysis/plots/{relativity}_{level}_{plottype}_plot.svg'},
        'timings_plot_png': {'type': 'file', 'mask': './timings_analysis/plots/{relativity}_{level}_{plottype}_plot.png'},
    }


def build_timings_df(rows, df_item, plottypes: dict[str, dict], **const_keys):
    for plottype, settings in plottypes.items():
        if not settings['accept'](**const_keys):
            continue
        df_item.include_element(pd.DataFrame([
            { **obj, **keys }
            for obj, keys in rows
        ]), plottype=plottype)


def unpack_relenergy_threshold(refmethod: str):
    def func(df: pd.DataFrame) -> pd.DataFrame:
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

        df['time_per_conf'] = df['time'] / (df['num_succ'] - df['num_above_relenergy'])

        refmethod_df = df[df['method'] == refmethod]
        refmethod_df = refmethod_df.rename(
            columns={'time_per_conf': 'refmethod_time_per_conformer', 'time': 'refmethod_time'}
        )[
            ['testcase', 'refmethod_time_per_conformer', 'refmethod_time', 'relenergy_threshold']
        ]
        df = df[~(df['method'] == refmethod)]
        df = df.merge(refmethod_df, on=['testcase', 'relenergy_threshold'], how='left')
        return df
    return func


def gen_timeplotting_settings(plottype: str, plot_modes: dict[str, any], result_settings) -> dict[str, any]:
    result_settings.include_element({
        key: value
        for pattern, cur_settings in plot_modes.items()
        if fnmatch.fnmatch(plottype, pattern)
        for key, value in cur_settings.items()
    })


def load_additional_data(df, df_result, testset_summary, settings_item):
    for df_obj, df_keys in df:
        plottype: str = df_keys['plottype']
        settings_obj = settings_item.access_element(plottype=plottype)
        
        df_obj = df_obj.copy()
        if 'use_data' in settings_obj:
            for key in settings_obj['use_data']:
                values = {
                    testcase_keys['testcase']: testcase_data[key]
                    for testcase_data, testcase_keys in testset_summary
                }
                df_obj[key] = df_obj['testcase'].apply(lambda x: values[x])
        df_result.include_element(df_obj, **df_keys)

def final_fix(df, thr):
    return df[df['relenergy_threshold'] == thr]

def timings_transforms(ds, main_logger) -> list[Transform]:
    REQUESTED_PLOTTYPES = ['basic-vsrdkit-doublelog']
    # REQUESTED_PLOTTYPES = ['basicdoftrash-low']
    # REQUESTED_PLOTTYPES = ['basic-doublelog']
    # REQUESTED_PLOTTYPES = ['basic-final-all', 'basic-final-low', 'basic-vsrdkit-doublelog', 'basic-doublelog']
    PLOT_MODES = {
        'basic-vsrdkit-doublelog': {
            'accept': lambda relativity, level, **kw: (relativity == 'global' and level == 'mmff'),
            'process': lambda df: unpack_relenergy_threshold(refmethod='ringo')(
                df[
                    df['method'].isin(['ringo', 'ETKDGv3-2024', 'ETKDGv1', 'mtd'])
                ]
            ),
            'general_caption': 'Time per unique\nconformer, s',
            'mcr_caption': 'Time per conformer by MCR, s',
            'alt_caption': 'Time per conformer by alternative method, s',
            'facet': {'v': 'method'},
            'show_legend': True,
        },
        'basic-final-all': {
            'accept': lambda relativity, level, **kw: (relativity == 'basic' and level == 'mmff'),
            'process': lambda df: final_fix(unpack_relenergy_threshold(refmethod='ringo')(
                df[
                    df['method'].isin(['ringo', 'ETKDGv3-2024', 'mtd', 'mmbasicOld', 'mmringOld', 'crestOld']) &
                    df['testcase'].isin(HEAVY_TESTCASES)
                ]
            ), thr='No E threshold'),
            'general_caption': 'Time per unique\nconformer, s',
            'mcr_caption': 'Time per conformer by MCR, s',
            'alt_caption': 'Time per conformer by alternative method, s',
            'facet': {'v': 'method'},
            'show_legend': True,
        },
        'basicdoftrash-low': {
            'accept': lambda relativity, level, **kw: (relativity == 'basic' and level == 'mmff'),
            'process': lambda df: df.assign(
                time_per_conf=df['time'] / (df['num_succ'] - df['num_above_relenergy_15'])
            ),
            'use_data': ['num_dofs'],
            'general_caption': 'Time per unique\nconformer, s',
            'mcr_caption': 'Time per conformer by MCR, s',
            'alt_caption': 'Time per conformer by alternative method, s',
            'facet': {'v': 'method'},
            'show_legend': True,
        },
        'basic-final-low': {
            'accept': lambda relativity, level, **kw: (relativity == 'basic' and level == 'mmff'),
            'process': lambda df: final_fix(unpack_relenergy_threshold(refmethod='ringo')(
                df[
                    df['method'].isin(['ringo', 'ETKDGv3-2024', 'mtd', 'mmbasicOld', 'mmringOld', 'crestOld']) &
                    df['testcase'].isin(HEAVY_TESTCASES)
                ]
            ), thr='E < 15 kcal/mol'),
            'use_data': ['num_dofs'],
            'general_caption': 'Time per unique\nconformer, s',
            'mcr_caption': 'Time per conformer by MCR, s',
            'alt_caption': 'Time per conformer by alternative method, s',
            'facet': {'v': 'method'},
            'show_legend': True,
        },

        'basic-doublelog': {
            'accept': lambda relativity, level, method, **kw: (relativity == 'basic' and level == 'mmff'),
            'process': unpack_relenergy_threshold(refmethod='ringo'),
            'general_caption': 'Time per unique\nconformer, s',
            'mcr_caption': 'Time per conformer by MCR, s',
            'alt_caption': 'Time per conformer by alternative method, s',
            'facet': {'h': 'relenergy_threshold', 'v': 'method'},
            'show_legend': True,
        },
        'globalmmff-doublelog': {
            'accept': lambda relativity, level, **kw: (relativity == 'global' and level == 'mmff'),
            'process': lambda df: unpack_relenergy_threshold(refmethod='ringo')(
                df[df['method'].isin(['ringo', 'ETKDGv3', 'ETKDGv3-2024', 'mtd', 'mmbasicOld', 'mmringOld', 'crestOld'])]
            ),
            'general_caption': 'Time per unique\nconformer, s',
            'mcr_caption': 'Time per conformer by MCR, s',
            'alt_caption': 'Time per conformer by alternative method, s',
            'facet': {'h': 'relenergy_threshold', 'v': 'method'},
            'show_legend': True,
        },

        # Older figure designs
        'low-*': {
            'accept': lambda relativity, level, **kw: (relativity == 'basic' and level == 'mmff'),
            'process': lambda df: df.assign(
                time_per_conf=df['time'] / (df['num_succ'] - df['num_above_relenergy_15'])
            ),
            'general_caption': 'Time per unique\nlow-energy conformer, s',
            'mcr_caption': 'Time per low-energy\nconformer by MCR, s',
            'alt_caption': 'Time per low-energy conformer by alternative method, s',
        },
        'any-*': {
            'process': lambda df: df.assign(
                time_per_conf=df['time'] / df['num_succ']
            ),
            'general_caption': 'Time per unique\nconformer, s',
            'mcr_caption': 'Time per conformer\nby MCR, s',
            'alt_caption': 'Time per conformer by alternative method, s',
        },
        '*-bar': {
            'show_x_labels': False,
            'show_legend': True,
            'log_time': True,
        },
        '*-dof': {
            'use_data': ['num_dofs'],
            'show_legend': True,
            'log_time': True,
        },
    }

    def preprocess_timings_df(df, settings):
        num_columns = [col for col in df.columns if col.startswith("num_")]
        df[num_columns] = df[num_columns].fillna(0).astype(int)
        df = settings['process'](df)

        # Cut the bar heights if required
        if 'time_cutoff' in settings:
            time_cutoff = settings['time_cutoff']
            df['time_per_conf'] = df['time_per_conf'].apply(lambda x: time_cutoff if x > time_cutoff else x)
        
        order_df = df.sort_values(by=['time_per_conf'], ascending=False).reset_index(drop=True)
        cat_type = CategoricalDtype(categories=order_df['testcase'].unique(), ordered=True)
        df['testcase'] = df['testcase'].astype(cat_type)
        df = df.reset_index(drop=True)
        return df

    def plot_timing_png(df, res_png, settings, plottype):
        time_caption = settings['mcr_caption']
        show_legend = settings['show_legend']
        # log_time = settings['log_time']

        if len(df) == 0:
            return None

        figure_size = (14, 10)
        if 'final' in plottype:
            figure_size = (8,8)
        # elif 'vsrdkit' in plottype:
        #     figure_size = (8, 5)

        from plotnine import ggplot, aes, labs, geom_abline, scale_x_log10, facet_grid, scale_fill_manual, geom_point, scale_y_log10, scale_color_manual, theme_bw, element_text, theme, element_blank, element_rect, element_line, geom_bar, position_dodge, ylim
        normal_theme = (theme_bw() +
            theme(
                panel_grid_major = element_blank(),
                panel_grid_minor = element_blank(),
                panel_border = element_rect(colour='black', fill=None, size=1),
                axis_line = element_line(colour='black'),
                axis_title = element_text(size=16, face='bold', ma='center'),
                axis_text = element_text(size=14),
                legend_title = element_text(size=14, face='bold'),
                legend_text = element_text(size=14),
                figure_size=figure_size,
                # figure_size=(10, 4),
            )
        )

        # Colors from Set2
        colorscheme = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf']
        # colorscheme = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#a65628", "#f781bf"]
        # colorscheme = ["#7570b3", "#66a61e"]

        df['method'] = df['method'].replace(METHOD_NAMES)
        covered_methods = df['method'].unique()
        method_type = CategoricalDtype(categories=[x for x in METHOD_ORDERING if x in covered_methods], ordered=True)
        df['method'] = df['method'].astype(method_type)

        # ic(df)
        # raise Exception("erghreuh")

        if plottype == 'basicdoftrash-low':
            return None

        if plottype.endswith('-bar'):
            plot = (
                ggplot(df, aes(y='time_per_conf', x='testcase', fill='method')) +
                geom_bar(stat='identity', position=position_dodge(width=0.9)) +
                labs(y=time_caption, x='Test macrocycle from BIRD') + normal_theme +
                theme(axis_text_x = element_text(angle = 60)) +
                scale_fill_manual(values=colorscheme)
            )
        elif plottype.endswith('-dof'):
            plot = (
                ggplot(df, aes(y='time_per_conf', x='num_dofs', color='method')) +
                geom_point(size=3) +
                labs(y=time_caption, x='Number of kinematic DOFs') + normal_theme +
                scale_color_manual(values=colorscheme)
            )
        elif plottype.endswith('-doublelog') or 'final' in plottype:
            plot = (
                ggplot(df, aes(x='time_per_conf', y='refmethod_time_per_conformer', color='method')) +
                normal_theme +
                geom_abline(intercept=0, slope=1, linetype='solid') +
                geom_abline(intercept=-1, slope=1, linetype='dashdot') +
                geom_abline(intercept=-2, slope=1, linetype='dashed') +
                geom_abline(intercept=-3, slope=1, linetype='dotted') +
                geom_point(size=2 if not 'final' in plottype else 3, shape='X') +
                scale_x_log10() +
                scale_y_log10() +
                labs(x = settings['alt_caption'], y = settings['mcr_caption']) +
                scale_color_manual(values=METHOD_COLORS) + facet_grid('method~relenergy_threshold')
            )
        else:
            raise Exception(f"Unknown plottype={plottype}")

        if 'time_cutoff' in settings:
            plot += ylim(0, settings['time_cutoff'])
        if not show_legend:
            plot += theme(legend_position="none")
        # if log_time:
        #     plot += scale_y_log10()
        if 'show_x_labels' in settings and not settings['show_x_labels']:
            plot += theme(axis_text_x=element_blank())
        plot.save(res_png, verbose=False)
        return res_png

    def svg_to_png(svg, png):
        from cairosvg import svg2png

        with open(svg, 'r') as f:
            svg_code = f.read()
        svg2png(bytestring=svg_code, write_to=png)
        return png

    timings_transforms: list[Transform] = [
        # Raw timings df
        templates.exec('build_timings_df',
            input='total_single_csdata_obj', output='timing_df',
            aware_keys=['relativity', 'level'], merged_keys=['method', 'timelimit', 'testset', 'testcase'],
            method=lambda total_single_csdata_obj, timing_df, **kw:
                build_timings_df(
                    rows=total_single_csdata_obj, df_item=timing_df,
                    plottypes={k: v for k, v in PLOT_MODES.items() if k in REQUESTED_PLOTTYPES},
                    **kw
                )
        ),
        templates.pd_to_csv('dump_raw_timing_df',
            input='timing_df', output='timings_raw_csv', index=False, sep=';'
        ),
        templates.pd_from_csv('load_raw_timing_df_from_checkpoint',
            input='timings_raw_csv', output='timing_df', sep=';'
        ),

        # Testset summaries
        templates.exec('gen_timeplotting_settings',
            input='timing_df', output='timing_plot_settings', aware_keys=['plottype'], merged_keys=['relativity', 'level'],
            method=lambda timing_plot_settings, plottype, **kw:
                gen_timeplotting_settings(
                    plot_modes=PLOT_MODES,
                    plottype=plottype,
                    result_settings=timing_plot_settings,
                )
        ),
        templates.exec('load_additional_data',
            input=['timing_df', 'final_summary_object', 'timing_plot_settings'], output='timing_df_with_data',
            merged_keys=['relativity', 'level', 'testset', 'testcase', 'plottype'],
            method=lambda timing_df, timing_df_with_data, final_summary_object, timing_plot_settings, **kw:
                load_additional_data(
                    df=timing_df,
                    df_result=timing_df_with_data,
                    testset_summary=final_summary_object,
                    settings_item=timing_plot_settings
                )
        ),

        templates.map('process_timing_df',
            input=['timing_df_with_data', 'timing_plot_settings'], output='timing_df_processed',
            aware_keys=['relativity', 'level'],
            mapping=lambda timing_df_with_data, timing_plot_settings, **kw:
                preprocess_timings_df(df=timing_df_with_data, settings=timing_plot_settings)
        ),
        templates.pd_to_csv('dump_final_timing_df',
            input='timing_df_processed', output='timings_final_csv', index=False, sep=';'
        ),
        templates.pd_from_csv('load_final_timing_df',
            input='timings_final_csv', output='timing_df_processed', sep=';'
        ),
        templates.map('plot_timings_to_svg',
            input=['timing_df_processed', 'timing_plot_settings'], output='timings_plot_svg',
            aware_keys=['relativity', 'level', 'plottype'],
            mapping=lambda timing_df_processed, timings_plot_svg, timing_plot_settings, plottype, **kw:
                plot_timing_png(
                    df=timing_df_processed,
                    res_png=timings_plot_svg,
                    settings=timing_plot_settings,
                    plottype=plottype
                ),
            include_none=False
        ),
        templates.map('timings_plot_to_png',
            input='timings_plot_svg', output='timings_plot_png',
            mapping=lambda timings_plot_svg, timings_plot_png: svg_to_png(svg=timings_plot_svg, png=timings_plot_png)
        )
    ]
    return timings_transforms
