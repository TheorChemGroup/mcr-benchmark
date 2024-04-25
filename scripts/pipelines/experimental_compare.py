import copy
import pandas as pd
from pandas.api.types import CategoricalDtype

from pysquared import Transform
import pysquared.transforms.transform_templates as templates
from utils import confsearch

from .diversity import select_clustered_elements
from .timings import METHOD_NAMES, HEAVY_TESTCASES, METHOD_COLORS


def expcompare_dataitems():
    return {
        # Compare with exp
        'expconformers_for_expcompare': {'type': 'object', 'keys': ['relativity', 'level', 'testset', 'testcase', 'method', 'timelimit']},
        'ensembles_for_expcompare': {'type': 'object', 'keys': ['relativity', 'level', 'testset', 'testcase', 'method', 'timelimit']},
        'exp_compare_result_obj': {'type': 'object', 'keys': ['relativity', 'level', 'testset', 'testcase', 'method', 'timelimit']},
        'exp_compare_result': {'type': 'file', 'mask': './exp_compare/data/{relativity}_{level}/{method}_{timelimit}/{testset}_{testcase}.json'},
        'expcompare_none_items': {'type': 'object', 'keys': ['relativity', 'level', 'testset', 'testcase', 'method', 'timelimit']},
        'total_expcompare_result_obj': {'type': 'object', 'keys': ['relativity', 'level', 'testset', 'testcase', 'method', 'timelimit']},
        'csdata_for_expcompare': {'type': 'object', 'keys': ['relativity', 'level', 'testset', 'testcase', 'method', 'timelimit']},
        'total_expcompare_dataitems': {'type': 'object', 'keys': ['relativity', 'level', 'testset', 'testcase', 'method', 'timelimit']},
        
        'compare_df': {'type': 'object', 'keys': ['plottype']},
        'compare_df_csv': {'type': 'file', 'mask': './exp_compare/dataframes/{plottype}_df.csv'},
        'compare_summary': {'type': 'object', 'keys': ['plottype']},
        'compare_summary_csv': {'type': 'file', 'mask': './exp_compare/{plottype}_summary.csv'},
        'compare_summary_plot_svg': {'type': 'file', 'mask': './exp_compare/plots/{plottype}_plot.svg'},
        'compare_summary_plot_png': {'type': 'file', 'mask': './exp_compare/plots/{plottype}_plot.png'},
    }


PLOTTING_MODES = {
    'fastmethods-allmols': {
        'accept': lambda relativity, method, level, testcase, **kw: (
            relativity == 'basic' and
            method in ('ringo', 'ETKDGv3-2024', 'mtd') and
            level == 'mmff'
        ),
        'method_names': METHOD_NAMES,
        'method_ordering': [
            'ringo',
            'ETKDGv3-2024',
            'mtd',
        ],
        'targets': ('table', 'plot'),
    },
    'allmethods-heavymols': {
        'accept': lambda relativity, method, testcase, level, **kw: (
            relativity == 'global' and
            method in ('ringo', 'ETKDGv3-2024', 'mtd', 'mmbasicOld', 'mmringOld', 'crestOld') and
            testcase in HEAVY_TESTCASES and
            level == 'mmff'
        ),
        'method_ordering': [
            'ringo',
            'ETKDGv3-2024',
            'mtd',
            'crestOld',
            'mmringOld',
            'mmbasicOld',
        ],
        'method_names': METHOD_NAMES,
        'targets': ('table',),
    },
    'mcr-vs-rdkit-ez': {
        'accept': lambda relativity, method, level, testcase, **kw: (
            relativity == 'global' and
            method in ('ETKDGv3-2024', 'ETKDGv1', 'ringo') and
            level == 'mmff'
        ),
        'method_names': METHOD_NAMES,
        'method_ordering': ['ringo', 'ETKDGv3-2024', 'ETKDGv1'],
        'targets': ('plot', 'table'),
    },
    'mcr-vs-rdkit': {
        'accept': lambda relativity, method, level, testcase, **kw: (
            relativity == 'global' and
            method in ('ringo-vs-rdkit2024', 'ETKDGv3-2024', 'ETKDGv1', 'ringo') and
            level == 'mmff'
        ),
        'method_names': METHOD_NAMES,
        'method_ordering': ['ringo', 'ringo-vs-rdkit2024', 'ETKDGv3-2024', 'ETKDGv1'],
        'targets': ('table', 'plot'),
    },
    'mcr-vs-mtd': {
        'accept': lambda relativity, method, level, testcase, **kw: (
            relativity == 'global' and
            method in ('ringo-vs-mtd', 'mtd',) and
            level == 'mmff'
        ),
        'method_names': METHOD_NAMES,
        'method_ordering': ['ringo-vs-mtd', 'mtd',],
        'targets': ('table', 'plot'),
    },
    'mcr-vs-crest': {
        'accept': lambda relativity, method, level, testcase, **kw: (
            testcase in HEAVY_TESTCASES and
            relativity == 'global' and
            method in ('ringo-vs-crest', 'ringo', 'crestOld')
        ),
        'method_names': METHOD_NAMES,
        'method_ordering': ['ringo-vs-crest', 'crestOld', 'ringo'],
        'targets': ('table',),
    },
    'mcr-vs-mmring': {
        'accept': lambda relativity, method, level, testcase, **kw: (
            relativity == 'global' and
            method in ('ringo-vs-mmring', 'ringo', 'mmringOld')
        ),
        'method_names': METHOD_NAMES,
        'method_ordering': ['ringo-vs-mmring', 'mmringOld', 'ringo'],
        'targets': ('table',),
    },
    'final-paper': {
        'accept': lambda relativity, method, level, testcase, **kw: (
            relativity == 'alllimit'
        ),
        'method_names': METHOD_NAMES,
        'method_ordering': ['ringo', 'ETKDGv3-2024', 'mtd', 'mmbasicOld', 'mmringOld', 'crestOld'],
        'targets': ('table', 'plot'),
        'timelimit_split': True,
    },
}


def compare_with_experimental():
    import sys
    import os
    import json
    module_dir = INSERT_HERE
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    os.chdir(os.path.dirname(__file__))
    
    import networkx as nx
    import ringo
    from utils.confsearch import ConformerInfo

    input_xyz_path = INSERT_HERE
    assert input_xyz_path == sys.argv[1]
    assert os.path.isfile(input_xyz_path)

    experimental_xyz_path = INSERT_HERE
    assert experimental_xyz_path == sys.argv[2]
    assert os.path.isfile(experimental_xyz_path)

    result_json_path = INSERT_HERE
    assert result_json_path == sys.argv[3]

    p = ringo.Confpool()
    p.include_from_file(experimental_xyz_path)
    assert len(p) == 1
    p.include_from_file(input_xyz_path)
    assert len(p) > 1, f"XYZ-file '{input_xyz_path}' is empty"
    
    p.generate_connectivity(0, mult=1.3)
    graph = p.get_connectivity().copy()
    all_nodes = [i for i in graph.nodes]
    bridges = list(nx.bridges(graph))
    graph.remove_edges_from(bridges)
    # Some of these connected components will be out cyclic parts, others are just single atoms
    components_lists = [list(comp) for comp in nx.connected_components(graph)]

    # Compute separate RMSD lists with respect to each cyclic part
    rmsd_lists = []
    for conn_component in components_lists:
        if len(conn_component) == 1:
            continue

        p.generate_connectivity(
            0, mult=1.3,
            ignore_elements=[
                node
                for node in all_nodes 
                if node not in conn_component
            ]
        )
        cur_graph = p.get_connectivity()
        assert cur_graph.number_of_nodes() == len(conn_component)
        p.generate_isomorphisms()
        rmsd_list = []
        for i in range(1, len(p)):
            rmsd, _, _ = p[0].rmsd(p[i])
            rmsd_list.append(rmsd)
        rmsd_lists.append(rmsd_list)

    crmsd_values = [max(x) for x in zip(*rmsd_lists)]
    min_index, min_crmsd_difference = min(enumerate(crmsd_values), key=lambda x: x[1])
    min_index += 1 # To get the index in 'p'
    conf_name = ConformerInfo(description=p[min_index].descr).data['name']

    res_data = {
        'min_crmsd': min_crmsd_difference,
        'conf_name': conf_name,
        'conf_index': min_index - 1, # Count the ref conformer out
    }
    with open(result_json_path, 'w') as f:
        json.dump(res_data, f)


def add_placeholders_to_expcompare_result_obj(input_obj, ref_data, result_obj):
    hold_keys = ['testset', 'testcase']

    covered_base_keys = [
        {k: v for k, v in keys.items() if k not in hold_keys}
        for _, keys in input_obj
    ]

    for _, ref_keys in ref_data:
        base_keys = {k: v for k, v in ref_keys.items() if k not in hold_keys}
        if base_keys not in covered_base_keys:
            continue

        if input_obj.contains_keys(keys=ref_keys):
            result_obj.include_element(
                copy.deepcopy(input_obj.access_element(**ref_keys)),
                **ref_keys
            )
        else:
            result_obj.include_element(
                None,
                **ref_keys
            )


def build_compare_df(rows, df, modes: dict[str, any]) -> None:
    for plottype, settings in modes.items():
        df_obj = pd.DataFrame([
            { **obj, **keys }
            for obj, keys in rows
            if settings['accept'](**keys)
        ])
        num_columns = [col for col in df_obj.columns if col.startswith("num_")]
        df_obj[num_columns] = df_obj[num_columns].fillna(0).astype(int)
        df_obj['timelimit'] = df_obj['timelimit'].astype(str)
        df.include_element(df_obj, plottype=plottype)


def get_expcompare_summary(df: pd.DataFrame, settings: dict[str, any]) -> pd.DataFrame:
    if 'table' not in settings['targets']:
        return None
    
    summary_df = pd.DataFrame(columns=['method', 'count', 'rmsd_threshold', 'timelimit', 'level'])
    timelimits = df['timelimit'].unique()
    for timelimit in timelimits:
        for rmsd_threshold in (0.5, 1.0):
            summary_part_df = (
                df[
                    (df['min_crmsd'] < rmsd_threshold) &
                    (df['timelimit'] == timelimit)
                ]
                .groupby(['method', 'level'])
                .size()
                .reset_index(name='count')
            )
            summary_part_df['rmsd_threshold'] = rmsd_threshold
            summary_part_df['timelimit'] = timelimit
            summary_df = (
                pd.concat([summary_df, summary_part_df])
                .reset_index(drop=True)
            )

    summary_df = summary_df.sort_values(by=['timelimit', 'method', 'rmsd_threshold'], ascending=True)
    ic(summary_df)
    return summary_df


def make_expcompare_plot(df: pd.DataFrame, res_png: str, settings: dict[str, any]) -> str:
    if 'plot' not in settings['targets']:
        return None
    
    timelimit_split = 'timelimit_split' in settings and settings['timelimit_split']
    
    from plotnine import ggplot, aes, labs, geom_histogram, xlim, coord_cartesian, after_stat, scale_fill_manual, geom_density, facet_grid, geom_point, scale_y_log10, scale_color_manual, theme_bw, element_text, theme, element_blank, element_rect, element_line, geom_bar, position_dodge, ylim
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
            figure_size=(10, 4)
        )
    )
    
    ADJUST_FACTOR = 0.3

    df['method'] = df['method'].replace(settings['method_names'])
    represented_methods = df['method'].unique()
    method_column_type = CategoricalDtype(categories=[
        settings['method_names'][method]
        for method in settings['method_ordering']
        if settings['method_names'][method] in represented_methods
    ], ordered=True)
    df['method'] = df['method'].astype(method_column_type)

    if not timelimit_split:
        plot = (
            ggplot(df,
            aes(x='min_crmsd', fill='method')) +
            normal_theme +
            geom_density(aes(y=after_stat("count")), alpha=0.3, adjust=ADJUST_FACTOR) +
            facet_grid('timelimit~.') +
            scale_fill_manual(values=METHOD_COLORS) +
            theme(figure_size=(8, 5)) +
            labs(y='Distribution density', x='CRMSD w.r.t. experimental macrocycle')
        )
    else:
        df['facet_col'] = df['method'].astype(str) + '_' + df['timelimit']

        plot = (
            ggplot(df,
            aes(x='min_crmsd', fill='method')) +
            normal_theme +
            geom_histogram(position='identity', alpha=1.0, binwidth=0.05) +
            # geom_density(aes(y=after_stat("count")), alpha=1.0, adjust=ADJUST_FACTOR) +
            facet_grid('facet_col~.') +
            scale_fill_manual(values=METHOD_COLORS) +
            theme(figure_size=(9, 10)) +
            labs(y='Distribution density', x='CRMSD w.r.t. experimental macrocycle') +
            xlim(0, 2.5)# + coord_cartesian(xlim=(0, 2.5))
        )

    plot.save(res_png, verbose=False)
    return res_png


def svg_to_png(svg_path: str, png_path: str) -> str:
    from cairosvg import svg2png

    with open(svg_path, 'r') as f:
        svg_code = f.read()
    svg2png(bytestring=svg_code, write_to=png_path)
    return png_path


def expcompare_transforms(ds, main_logger) -> list[Transform]:

    expcompare_transforms: list[Transform] = [
        # Generate comparison data
        templates.exec('restrict_opt_experimental_conformers',
            input=['final_ensemble_path', 'final_expconformers_path'], output='expconformers_for_expcompare',
            merged_keys=['relativity', 'level', 'testset', 'testcase', 'method', 'timelimit'],
            method=lambda final_ensemble_path, final_expconformers_path, expconformers_for_expcompare:
                select_clustered_elements(
                    all_elements=final_expconformers_path,
                    ref_item=final_ensemble_path,
                    selected_elements=expconformers_for_expcompare,
                    ignore_if_missing=True
                )
        ),
        templates.exec('restrict_ensembles_for_expcompare',
            input=['final_ensemble_path', 'expconformers_for_expcompare'], output='ensembles_for_expcompare',
            merged_keys=['relativity', 'level', 'testset', 'testcase', 'method', 'timelimit'],
            method=lambda final_ensemble_path, expconformers_for_expcompare, ensembles_for_expcompare:
                select_clustered_elements(
                    all_elements=final_ensemble_path,
                    ref_item=expconformers_for_expcompare,
                    selected_elements=ensembles_for_expcompare
                )
        ),
        templates.pyfunction_subprocess('compare_with_experimental',
            input=['ensembles_for_expcompare', 'expconformers_for_expcompare'], output='exp_compare_result',
            pyfunction=compare_with_experimental, calcdir='calcdir', nproc=1,
            argv_prepare=lambda ensembles_for_expcompare, expconformers_for_expcompare, exp_compare_result, **kw:
                (ensembles_for_expcompare.access_element(), expconformers_for_expcompare.access_element(), exp_compare_result.get_path()),
            subs=lambda ensembles_for_expcompare, expconformers_for_expcompare, exp_compare_result, **kw: {
                'input_xyz_path': ensembles_for_expcompare.access_element(),
                'experimental_xyz_path': expconformers_for_expcompare.access_element(),
                'result_json_path': exp_compare_result.get_path(),
            },
            output_process=lambda exp_compare_result, **kw: confsearch.assertive_include(exp_compare_result)
        ),

        # Create complete comparison data
        templates.load_json('load_expcompare_results', input='exp_compare_result', output='exp_compare_result_obj'),
        templates.exec('add_placeholders_to_expcompare_result_obj',
            input=['exp_compare_result_obj', 'total_single_csdata_obj'], output='total_expcompare_result_obj',
            merged_keys=['relativity', 'level', 'testset', 'testcase', 'method', 'timelimit'],
            method=lambda exp_compare_result_obj, total_single_csdata_obj, total_expcompare_result_obj:
                add_placeholders_to_expcompare_result_obj(
                    input_obj=exp_compare_result_obj,
                    ref_data=total_single_csdata_obj,
                    result_obj=total_expcompare_result_obj,
                )
        ),
        templates.restrict('finalize_placeholder_expcompare_none_items',
            input='total_single_csdata_obj', ref='total_expcompare_result_obj', output='csdata_for_expcompare',
            merged_keys=['relativity', 'level', 'testset', 'testcase', 'method', 'timelimit']
        ),
        templates.map('build_expcompare_dataitems',
            input=['total_expcompare_result_obj', 'csdata_for_expcompare'], output='total_expcompare_dataitems',
            mapping=lambda total_expcompare_result_obj, csdata_for_expcompare, **kw: {
                    **csdata_for_expcompare,
                    'min_crmsd': total_expcompare_result_obj['min_crmsd']
                        if total_expcompare_result_obj is not None else None
                }
        ),
        templates.exec('build_compare_df',
            input='total_expcompare_dataitems', output='compare_df',
            merged_keys=['relativity', 'level', 'method', 'timelimit', 'testset', 'testcase'],
            method=lambda total_expcompare_dataitems, compare_df, **kw:
                build_compare_df(rows=total_expcompare_dataitems, df=compare_df,
                    modes={k: v for k, v in PLOTTING_MODES.items() if k in (
                        # 'fastmethods-allmols',
                        # 'allmethods-heavymols',
                        'mcr-vs-rdkit-ez',
                        'mcr-vs-rdkit',
                        # 'mcr-vs-mtd',
                        # 'mcr-vs-crest',
                        # 'final-paper',
                    )
                })
        ),
        templates.pd_to_csv('dump_compare_df',
            input='compare_df', output='compare_df_csv', index=False, sep=';'
        ),
        templates.pd_from_csv('load_compare_df',
            input='compare_df_csv', output='compare_df', sep=';'
        ),

        # Create compare summary table
        templates.map('get_expcompare_summary',
            input='compare_df', output='compare_summary', aware_keys=['plottype'],
            mapping=lambda compare_df, plottype, **kw:
                get_expcompare_summary(compare_df, settings=PLOTTING_MODES[plottype]),
            include_none=False
        ),
        templates.pd_to_csv('dump_compare_summary',
            input='compare_summary', output='compare_summary_csv', index=False, sep=';'
        ),

        # Create plots
        templates.map('make_expcompare_plot',
            input='compare_df', output='compare_summary_plot_svg', aware_keys=['plottype'],
            mapping=lambda compare_df, compare_summary_plot_svg, plottype, **kw:
                make_expcompare_plot(df=compare_df, res_png=compare_summary_plot_svg, settings=PLOTTING_MODES[plottype]),
            include_none=False
        ),
        templates.map('expcompare_plot_to_png',
            input='compare_summary_plot_svg', output='compare_summary_plot_png',
            mapping=lambda compare_summary_plot_svg, compare_summary_plot_png:
                svg_to_png(svg_path=compare_summary_plot_svg, png_path=compare_summary_plot_png)
        )
    ]
    return expcompare_transforms
