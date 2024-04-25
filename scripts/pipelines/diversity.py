import datetime
import pandas as pd
import numpy as np

from pysquared import Transform
import pysquared.transforms.transform_templates as templates

from .timings import METHOD_NAMES
from .energy_distribution import MCR_VS_RDKIT_METHODS
from utils import confsearch


def diversity_dataitems():
    return {
        'diversity_ensemble_xyz_paths': {'type': 'object', 'keys': ['relativity', 'level', 'testset', 'testcase', 'method', 'timelimit', 'divplot_type']},
        'ensemble_relenergies_json_for_diversity': {'type': 'object', 'keys': ['relativity', 'level', 'testset', 'testcase', 'method', 'timelimit', 'divplot_type']},
        'initial_diversity_ensemble': {'type': 'file', 'mask': './diversity/ensembles/initial/{relativity}_{level}_{divplot_type}/{method}_{timelimit}/{testset}_{testcase}.xyz'},
        'merged_diversity_ensemble': {'type': 'file', 'mask': './diversity/ensembles/merged/{relativity}_{level}_{divplot_type}/{testset}_{testcase}.xyz'},
        'crmsd_matrix': {'type': 'file', 'mask': './diversity/matrices/{relativity}_{level}_{divplot_type}/{testset}_{testcase}.rmsd'},
        'final_diversity_ensemble': {'type': 'file', 'mask': './diversity/ensembles/perspective/{relativity}_{level}_{divplot_type}/{testset}_{testcase}.xyz'},
        
        'clustering_indices_json': {'type': 'file', 'mask': './diversity/clusterings/indices/{relativity}_{level}_{divplot_type}/{testset}_{testcase}.json'},
        'clustering_reachability_plot': {'type': 'file', 'mask': './diversity/clusterings/rplots/{relativity}_{level}_{divplot_type}/{testset}_{testcase}_png.zip'},
        'bestclustering_indices_json': {'type': 'file', 'mask': './diversity/bestclusterings/indices/{relativity}_{level}_{divplot_type}/{testset}_{testcase}.json'},
        'bestclustering_reachability_plot': {'type': 'file', 'mask': './diversity/bestclusterings/rplots/{relativity}_{level}_{divplot_type}/{testset}_{testcase}.png'},

        'clustered_crmsd_matrix_paths': {'type': 'object', 'keys': ['relativity', 'level', 'testset', 'testcase', 'divplot_type']},
        'clustered_final_ensemble_paths': {'type': 'object', 'keys': ['relativity', 'level', 'testset', 'testcase', 'divplot_type']},
        'cluster_medoid_indices_json': {'type': 'file', 'mask': './diversity/bestclusterings/medoids/{relativity}_{level}_{divplot_type}/{testset}_{testcase}.json'},
        'embedding_2d_json': {'type': 'file', 'mask': './diversity/embeddings/{relativity}_{level}_{divplot_type}/{testset}_{testcase}.json'},

        'point_dfs': {'type': 'object', 'keys': ['relativity', 'level', 'testset', 'testcase', 'divplot_type']},
        'cluster_dfs': {'type': 'object', 'keys': ['relativity', 'level', 'testset', 'testcase', 'divplot_type']},
        'relevant_stats_objs': {'type': 'object', 'keys': ['relativity', 'level', 'testset', 'testcase', 'method', 'timelimit', 'divplot_type']},
        'timing_info_raw': {'type': 'object', 'keys': ['relativity', 'level', 'testset', 'testcase', 'method', 'divplot_type']},
        'timing_info': {'type': 'object', 'keys': ['relativity', 'level', 'testset', 'testcase', 'divplot_type']},
        'diversity_dfs': {'type': 'object', 'keys': ['relativity', 'level', 'testset', 'testcase', 'divplot_type']},
        'point_df_csvs': {'type': 'file', 'mask': './diversity/dfs/point/{relativity}_{level}_{divplot_type}/{testset}_{testcase}.csv'},
        'cluster_df_csvs': {'type': 'file', 'mask': './diversity/dfs/cluster/{relativity}_{level}_{divplot_type}/{testset}_{testcase}.csv'},
        'diversity_df_csvs': {'type': 'file', 'mask': './diversity/dfs/diversity/{relativity}_{level}_{divplot_type}/{testset}_{testcase}.csv'},

        'diversity_plot_svg': {'type': 'file', 'mask': './diversity/plots/diversity/{relativity}_{level}_{divplot_type}/{testset}_{testcase}.svg'},
        'diversity_plot_png': {'type': 'file', 'mask': './diversity/plots/diversity/{relativity}_{level}_{divplot_type}/{testset}_{testcase}.png'},
        'diversity_summary_pdf': {'type': 'file', 'mask': './diversity/reports/{relativity}_{level}_{divplot_type}.pdf'},

        # Verification that all clusters are discovered by MCR with CREST runtimes
        'diversity_crosscompare_data': {'type': 'object', 'keys': ['relativity', 'level', 'testset', 'testcase', 'divplot_type']},
        'diversity_crosscompare_json': {'type': 'file', 'mask': './diversity/plots/diversity/crosscompare/inputpaths/{relativity}_{level}_{divplot_type}/{testset}_{testcase}.json'},
        'diversity_crosscompare_results_json': {'type': 'file', 'mask': './diversity/plots/diversity/crosscompare/result/{relativity}_{level}_{divplot_type}/{testset}_{testcase}.json'},
        'diversity_crosscompare_results': {'type': 'object', 'keys': ['relativity', 'level', 'testset', 'testcase', 'divplot_type']},
        'diversity_crosscompare_analysis': {'type': 'object', 'keys': ['relativity', 'level', 'testset', 'testcase', 'divplot_type']},
        'diversity_crosscompare_analysis_csv': {'type': 'file', 'mask': './diversity/plots/diversity/crosscompare/result/{relativity}_{level}_{divplot_type}/{testset}_{testcase}.csv'},
        'diversity_crosscompare_summary': {'type': 'object', 'keys': ['relativity', 'level', 'divplot_type']},
        'diversity_crosscompare_summary_csv': {'type': 'file', 'mask': './diversity/plots/diversity/crosscompare/result/summaries/{relativity}_{level}_{divplot_type}.csv'},
    }

def get_generic_parameters(**settings):
    return {
        'parameters': {
                # Run CRMSD filtering for very large ensembles
                'use_crmsd_filter': False,

                # Energy cutoff to define low-energy or perspective conformers
                'energy_cutoff': 15, # kcal/mol in relenergy
                'perspective_crmsd_cutoff': 0.2,

                'expand_clusters': True,
                'separate_unclustered': False,

                # Method names
                'base_method': 'ringo',
                'method_names': METHOD_NAMES,
                'method_labels': {
                    'ringo': 'Ringo (only)',
                    'other': 'Alternative (only)',
                    'both': 'Both',
                    'ringo_unclustered': 'Ringo (unclustered)',
                    'other_unclustered': 'Alternative (unclustered)',
                    'other_explored': 'Space explored by alternative',
                    'ringo_explored': 'Space explored by MCR',
                    'other_explored_exclusive': 'Space explored only by alternative',
                    'ringo_explored_exclusive': 'Space explored only by MCR',
                },
                'methodtype_ordering': [
                    'ringo',
                    'both',
                    'other',
                ],

                # Diversity plot settings
                'show_alternative_all': False,
                'show_alternative_exclusive': True,
                'show_mcr_exclusive': True,
                'show_times_slower': False,
                'show_exact_alttime': True,
                **settings,
            }
    }

DIVERSITY_ANALYSIS_MODES = {
    # 'basic': {
    #     'accept_xyz': lambda relativity, **kw: relativity == 'basic',
    #     **get_generic_parameters(),
    # },
    # 'vsrdkit': {
    #     'accept_xyz': lambda relativity, level, method, **kw: (
    #         relativity == 'global' and
    #         level == 'mmff' and
    #         method in ('ringo-vs-rdkit', 'ETKDGv3', 'ETKDGv1')
    #     ),
    #     **get_generic_parameters(base_method='ringo-vs-rdkit'),
    # },
    # 'vsmtd': {
    #     'accept_xyz': lambda relativity, level, method, **kw: (
    #         relativity == 'global' and
    #         level == 'mmff' and
    #         method in ('ringo-vs-mtd', 'mtd')
    #     ),
    #     **get_generic_parameters(base_method='ringo-vs-mtd'),
    # },

    'basicOFive': {
        'accept_xyz': lambda relativity, **kw: relativity == 'basic',
        **get_generic_parameters(perspective_crmsd_cutoff=0.5),
    },
    'vsrdkitOFiveNew': {
        'accept_xyz': lambda relativity, level, method, **kw: (
            relativity == 'global' and
            level == 'mmff' and
            method in ('ringo-vs-rdkit2024', 'ETKDGv3-2024', 'ETKDGv1')
        ),
        **get_generic_parameters(base_method='ringo-vs-rdkit2024', perspective_crmsd_cutoff=0.5),
    },
    'vsmtdOFive': {
        'accept_xyz': lambda relativity, level, method, **kw: (
            relativity == 'global' and
            level == 'mmff' and
            method in ('ringo-vs-mtd', 'mtd')
        ),
        **get_generic_parameters(base_method='ringo-vs-mtd', perspective_crmsd_cutoff=0.5),
    },
    # 'mcr-vs-crest-large': {
    #     'accept_xyz': lambda method, timelimit, level: (
    #         method in ('crestOld', 'ringo-vs-crest') and
    #         timelimit == 'long' and
    #         level == 'mmff'
    #     ),

    #     'parameters': {
    #         # Run CRMSD filtering for very large ensembles
    #         'use_crmsd_filter': True,
    #         'crmsd_filter_cutoff': 0.2,

    #         # Energy cutoff to define low-energy or perspective conformers
    #         'energy_cutoff': 15, # kcal/mol in relenergy

    #         'expand_clusters': True,
    #         'separate_unclustered': False,

    #         # Method names
    #         'base_method': 'ringo',
    #         'method_names': METHOD_NAMES,
    #         'method_labels': {
    #             'ringo': 'Ringo (only)',
    #             'other': 'Alternative (only)',
    #             'both': 'Both',
    #             'ringo_unclustered': 'Ringo (unclustered)',
    #             'other_unclustered': 'Alternative (unclustered)',
    #             'other_explored': 'Space explored by alternative',
    #             'ringo_explored': 'Space explored by MCR',
    #             'other_explored_exclusive': 'Space explored only by alternative',
    #             'ringo_explored_exclusive': 'Space explored only by MCR',
    #         },
    #         'methodtype_ordering': [
    #             'ringo',
    #             'both',
    #             'other',
    #             # 'ringo_unclustered',
    #             # 'other_unclustered',
    #         ],

    #         # Diversity plot settings
    #         'show_alternative_all': False,
    #         'show_alternative_exclusive': True,
    #         'show_mcr_exclusive': True,
    #     },
    # },
}
for mode_item in DIVERSITY_ANALYSIS_MODES.values():
    parameters = mode_item['parameters']
    method_labels = parameters['method_labels']
    parameters['methodtype_ordering'] = [
        method_labels[method_name]
        for method_name in parameters['methodtype_ordering']
    ]


CRMSD_PYXYZ_SETTINGS = {
    'mirror_match': True,
    'print_status': False,
}
RPYTHON_ACTIVATION = """\
source /s/ls4/users/knvvv/mambaactivate
mamba activate rpyenv
"""


def load_diversity_ensemble_xyz_paths(xyz_path: str, **kw) -> None:
    return (
        (xyz_path, {'divplot_type': divplot_type})
        for divplot_type, settings in DIVERSITY_ANALYSIS_MODES.items()
        if settings['accept_xyz'](**kw)
    )


def prepare_initial_ensembles():
    import sys
    import os
    module_dir = INSERT_HERE
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    os.chdir(os.path.dirname(__file__))

    import ringo
    import json
    from utils.confsearch import (
        ConformerInfo,
        get_crmsd_matrix,
    )
    
    from icecream import install
    install()

    raw_xyz_path = INSERT_HERE
    assert os.path.exists(raw_xyz_path)
    relenergies_json = INSERT_HERE
    assert os.path.exists(relenergies_json)
    target_xyz_path = INSERT_HERE

    mode_settings = INSERT_HERE
    CRMSD_PYXYZ_SETTINGS = INSERT_HERE

    with open(relenergies_json, 'r') as f:
        relenergies_list = json.load(f)['relative']

    p = ringo.Confpool()
    p.include_from_file(raw_xyz_path)
    assert len(relenergies_list) == len(p), (
        f"Len p = {len(p)}, len(relenergies) = {len(relenergies_list)}"
    )

    for m, relenergy in zip(p, relenergies_list):
        m.descr = ConformerInfo.updated_status(
            old_descr=m.descr,
            relenergy=relenergy
        )

    before_filter_size = len(p)
    p.filter(lambda m: ConformerInfo(description=m.descr).is_successful)
    ic('Before/after fail filter', before_filter_size, len(p))
    
    if mode_settings['use_crmsd_filter']:
        p['index'] = lambda m: ConformerInfo(description=m.descr).index
        p['relenergy'] = lambda m: ConformerInfo(description=m.descr).relenergy
        p.sort('relenergy')
        before_crmsd_size = len(p)

        crmsd_matrix = get_crmsd_matrix(p, **CRMSD_PYXYZ_SETTINGS)
        ic(crmsd_matrix)
        p.rmsd_filter(mode_settings['crmsd_filter_cutoff'], rmsd_matrix=crmsd_matrix)
        ic('Before/after CRMSD prefilter', before_crmsd_size, len(p))
        p.sort('index')
    
    if len(p) > 0:
        p.save_xyz(target_xyz_path)


def merge_initial_ensembles():
    import sys
    import os
    import json
    module_dir = INSERT_HERE
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    os.chdir(os.path.dirname(__file__))

    import ringo
    from utils.confsearch import (
        ConformerInfo,
        get_crmsd_matrix,
    )
    
    from icecream import install
    install()

    input_xyz_paths = INSERT_HERE
    target_xyz_path = INSERT_HERE

    mode_settings = INSERT_HERE

    p = ringo.Confpool()
    for xyzname, keys in input_xyz_paths:
        method = keys['method']
        timelimit = keys['timelimit']

        before_size = len(p)
        assert os.path.exists(xyzname)
        p.include_from_file(xyzname)
        after_size = len(p)
        for i in range(before_size, after_size):
            info = ConformerInfo(description=p[i].descr)
            info.data['generated'] = {'method': method, 'timelimit': timelimit}
            p[i].descr = info.as_str()
    p.save_xyz(target_xyz_path)


def generate_crmsd_matrix():
    import sys
    import os
    module_dir = INSERT_HERE
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    os.chdir(os.path.dirname(__file__))

    import shutil
    import ringo
    import numpy as np
    from utils.confsearch import (
        get_crmsd_matrix,
        remove_unperspective_conformers,
    )
    
    from icecream import install
    install()

    xyz_path = INSERT_HERE
    assert os.path.exists(xyz_path)
    crmsd_matrix_path = INSERT_HERE
    target_xyz_path = INSERT_HERE

    mode_settings = INSERT_HERE
    CRMSD_PYXYZ_SETTINGS = INSERT_HERE

    p = ringo.Confpool()
    p.include_from_file(xyz_path)
    if len(p) == 0:
        sys.exit(0)
    ic(len(p))

    crmsd_matrix: np.ndarray = get_crmsd_matrix(p, **CRMSD_PYXYZ_SETTINGS)
    ic(crmsd_matrix)

    perspective_crmsd_matrix = remove_unperspective_conformers(
        p,
        crmsd_matrix=crmsd_matrix,
        lowenergy_threshold=mode_settings['energy_cutoff'],
        crmsd_cutoff=mode_settings['perspective_crmsd_cutoff']
    )
    if len(p) == 0:
        sys.exit(0)
    p.save_xyz(target_xyz_path)

    np.savez_compressed('tempfile', data=perspective_crmsd_matrix)
    shutil.move('tempfile.npz', crmsd_matrix_path)


def generate_all_clusterings():
    import sys
    import os
    module_dir = INSERT_HERE
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    os.chdir(os.path.dirname(__file__))

    import json
    import time
    import glob
    import shutil
    import zipfile
    import numpy as np
    import rpy2.robjects as robjects
    import rpy2.rinterface as rinterface
    
    from icecream import install
    install()

    crmsd_matrix_path = INSERT_HERE
    assert os.path.exists(crmsd_matrix_path)
    target_json_path = INSERT_HERE
    target_png_path = INSERT_HERE

    TEMP_PNG_DIR = './pngs'
    os.mkdir(TEMP_PNG_DIR)
    temp_png_path = os.path.join(TEMP_PNG_DIR, '{clustering_type}.png')

    mode_settings = INSERT_HERE

    rinterface.initr()

    # Uncompress the compressed RMSD matrix
    rmsd_matrix = np.load(crmsd_matrix_path)['data']
    temp_rmsd_txt = './tempmatrix.txt'
    np.savetxt(temp_rmsd_txt, rmsd_matrix)

    def tryexec_r(code):
        try:
            robjects.r("dev.off()")
        except:
            pass
        try:
            robjects.r(code)
        except:
            return False
        return True
    
    CLUSTERING_EPS_VALUES = tuple(c for c in np.arange(0.1, 0.7, 0.01))
    max_eps=2.0
    minpts=5
    cluster_labels = {}
    robjects.r(f"""\
sink("r_output.txt")
library(dbscan)
matrix_data <- as.dist(as.matrix(read.table("{temp_rmsd_txt}")))
res_full <- optics(matrix_data, eps={max_eps}, minPts = {minpts})
""")
    good = tryexec_r(f"""\
res <- extractXi(res_full, xi=0.05)
png("{temp_png_path.format(clustering_type='auto')}")
plot(res)
dev.off()
""")
    if good:
        try:
            cluster_labels['auto'] = [
                int(i)
                for i in list(robjects.r('res$cluster'))
            ]
        except:
            pass

    for cur_eps in CLUSTERING_EPS_VALUES:
        eps_round = round(cur_eps, 2)
        good = tryexec_r(f"""\
res <- extractDBSCAN(res_full, eps_cl = {cur_eps})
png("{temp_png_path.format(clustering_type=eps_round)}")
plot(res)
dev.off()
""")
        if good:
            try:
                cluster_labels[eps_round] = list(robjects.r('res$cluster'))
            except:
                pass
        time.sleep(1)
    robjects.r("sink()")

    with open(target_json_path, 'w') as f:
        json.dump(cluster_labels, f)
    
    with zipfile.ZipFile(target_png_path, 'w') as f:
        # Iterate over all the files in the directory and add them to the zip file
        for temp_png_path in glob.glob(os.path.join(TEMP_PNG_DIR, '*')):
            assert temp_png_path.endswith('.png')
            f.write(temp_png_path)

    os.remove(temp_rmsd_txt)
    shutil.rmtree(TEMP_PNG_DIR)


def find_best_clusterings():
    import sys
    import os
    module_dir = INSERT_HERE
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    os.chdir(os.path.dirname(__file__))

    import shutil
    import json
    import zipfile
    
    from icecream import install
    install()

    input_clustering_json = INSERT_HERE
    assert os.path.isfile(ic(input_clustering_json))
    input_clustering_pngs_zip = INSERT_HERE
    assert os.path.isfile(ic(input_clustering_pngs_zip))
    target_clustering_json = INSERT_HERE
    target_clustering_png = INSERT_HERE

    mode_settings = INSERT_HERE
    
    with open(input_clustering_json, 'r') as f:
        clustering_data = json.load(f)

    good_clusterings = []
    for cluster_type, cluster_indices in clustering_data.items():
        if cluster_type == 'auto':
            # cluster_indices = [i if i != 1 else 0 for i in cluster_indices]
            continue

        cluster_indices = [int(i) for i in cluster_indices]
        cluster_types = set(cluster_indices)
        cluster_types.discard(0) # Remove the "cluster" for all unclustered conformations

        num_confs = len(cluster_indices)
        ratio_unclustered = cluster_indices.count(0) / num_confs
        if ratio_unclustered > 0.8 or len(cluster_types) <= 2:
            continue
        
        cluster_sizes = sorted([cluster_indices.count(item)/num_confs for item in cluster_types], reverse=True)
        if cluster_sizes[0] - cluster_sizes[1] > 0.4 or cluster_sizes[1] - cluster_sizes[2] > 0.4:
            continue

        good_clusterings.append({
            'type': cluster_type,
            'score': -ratio_unclustered + len(cluster_types)/50,
        })

    if len(good_clusterings) == 0:
        ic("No good clusterings")
        sys.exit(0)

    best_clustering = max(good_clusterings, key=lambda item: item['score'])
    best_clustering_mode = best_clustering['type']
    
    with open(target_clustering_json, 'w') as f:
        json.dump(clustering_data[best_clustering_mode], f)
    
    desired_png = f'{best_clustering_mode}.png'
    file_location = os.path.join('pngs', desired_png)
    with zipfile.ZipFile(input_clustering_pngs_zip, 'r') as f:
        f.extract(file_location, '.')
    shutil.copy2(os.path.join('.', file_location), target_clustering_png)


def select_clustered_elements(all_elements, ref_item, selected_elements, ignore_if_missing=False):
    input_keys = set(all_elements.public_keys)
    ref_keys = set(ref_item.public_keys)
    output_keys = set(selected_elements.public_keys)
    restriction_set = ref_keys.intersection(output_keys)

    accepted_kvpairs = []
    for _, keys in ref_item:
        kvpair = {
            key: value
            for key, value in keys.items()
            if key in restriction_set
        }
        if kvpair not in accepted_kvpairs:
            accepted_kvpairs.append(kvpair)

    for accepted_kvpair in accepted_kvpairs:
        input_kv = {k: v for k, v in accepted_kvpair.items() if k in input_keys}
        if ignore_if_missing and not all_elements.contains_keys(keys=input_kv):
            continue
        selected_elements.include_element(
            all_elements.access_element(
                **input_kv
            ), **{k: v for k, v in accepted_kvpair.items() if k in output_keys}
        )


def generate_cluster_medoids():
    import sys
    import os
    module_dir = INSERT_HERE
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    os.chdir(os.path.dirname(__file__))

    import json
    import numpy as np
    
    from icecream import install
    install()

    crmsd_matrix_path = INSERT_HERE
    assert os.path.isfile(ic(crmsd_matrix_path))
    bestclustering_indices_json = INSERT_HERE
    assert os.path.isfile(ic(bestclustering_indices_json))
    cluster_medoid_indices_json = INSERT_HERE

    mode_settings = INSERT_HERE
    
    rmsd_matrix = np.load(crmsd_matrix_path)['data']
    with open(bestclustering_indices_json, 'r') as f:
        cluster_labels = json.load(f)
    
    clusters_set = set(cluster_labels)
    cluster_labels = np.array(cluster_labels)
    
    # Find medoid for each cluster
    medoid_idxs = {}
    for cluster_idx in clusters_set:
        mask = (cluster_labels == cluster_idx)
        rmsd_submatrix = rmsd_matrix[mask][:, mask]
        
        # get index map 'rmsd_submatrix' => 'rmsd_matrix'
        indices_map = np.where(mask)[0]
        # Calculate total distances
        total_distances = np.sum(rmsd_submatrix, axis=0)
        # Find medoid index
        medoid_index = np.argmin(total_distances) # index space of 'rmsd_submatrix'
        medoid_index = indices_map[medoid_index] # index space of 'rmsd_matrix'
        medoid_idxs[cluster_idx] = medoid_index

    # Get mask for generation of medoid RMSD submatrix
    cluster_idxs = sorted(list(clusters_set))
    expected_clusters = list(range(max(cluster_idxs) + 1))
    # Remove 0 cluster if there are no unclustered conformers
    if 0 not in cluster_idxs:
        del expected_clusters[expected_clusters.index(0)]
    assert cluster_idxs == expected_clusters, f"Indexing of clusters is broken."\
        f"Actual={repr(cluster_idxs)}\nExpected={expected_clusters}"
    medoid_idxs = np.array([medoid_idxs[i] for i in cluster_idxs])
    medoid_idxs = medoid_idxs.tolist()
    
    with open(cluster_medoid_indices_json, 'w') as f:
        json.dump(medoid_idxs, f)
    
    # Create medoid RMSD matrix = submatrix of 'rmsd_matrix'
    # medoid_rmsd_matrix = rmsd_matrix[medoid_idxs][:, medoid_idxs]
    # np.savez_compressed(medoid_rmsd_fname, data=medoid_rmsd_matrix)


def generate_2d_embeddings():
    import sys
    import os
    module_dir = INSERT_HERE
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    os.chdir(os.path.dirname(__file__))

    import json
    import numpy as np
    from sklearn.manifold import TSNE
    
    from icecream import install
    install()

    crmsd_matrix_path = INSERT_HERE
    assert os.path.isfile(ic(crmsd_matrix_path))
    embedding_2d_json_path = INSERT_HERE

    mode_settings = INSERT_HERE
    
    # Load the precomputed RMSD matrix
    rmsd_matrix = np.load(crmsd_matrix_path)['data']
    assert rmsd_matrix.shape[0] == rmsd_matrix.shape[1]

    # Obtain 2D embedding
    perplexity = 50
    if perplexity > rmsd_matrix.shape[0]:
        perplexity = rmsd_matrix.shape[0] - 1

    # Create an instance of the t-SNE algorithm
    tsne = TSNE(metric='precomputed', init='random', perplexity=perplexity, learning_rate=10, n_iter=2000)
    # Perform t-SNE embedding
    embedding_xy = tsne.fit_transform(rmsd_matrix)
    assert embedding_xy.shape[1] == 2

    embedding_xy_dict = {
        'x': [float(row[0]) for row in embedding_xy],
        'y': [float(row[1]) for row in embedding_xy],
    }
    
    with open(embedding_2d_json_path, 'w') as f:
        json.dump(embedding_xy_dict, f)


def generate_point_dfs():
    import sys
    import os
    module_dir = INSERT_HERE
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    os.chdir(os.path.dirname(__file__))

    import json
    import pandas as pd
    import ringo
    from utils.confsearch import (
        ConformerInfo,
        get_crmsd_matrix,
    )
    
    from icecream import install
    install()


    ensemble_xyz_path = INSERT_HERE
    assert os.path.isfile(ic(ensemble_xyz_path))
    clustering_json = INSERT_HERE
    assert os.path.isfile(ic(clustering_json))
    medoid_indices_path = INSERT_HERE
    assert os.path.isfile(ic(medoid_indices_path))
    embedding_2d_json = INSERT_HERE
    assert os.path.isfile(ic(embedding_2d_json))
    result_csv_path = INSERT_HERE
    ic(result_csv_path)

    mode_settings = INSERT_HERE
    
    p = ringo.Confpool()
    p.include_from_file(ensemble_xyz_path)

    data = [
        ConformerInfo(description=m.descr).data
        for m in p
    ]
    for row in data:
        delete_keys = []
        new_keys = {}
        for key, value in row.items():
            if isinstance(value, dict):
                assert key == 'generated', key
                for new_key, new_value in value.items():
                    new_keys[new_key] = new_value
                delete_keys.append(key)
        for key in delete_keys:
            del row[key]
        for key, value in new_keys.items():
            row[key] = value

    df = pd.DataFrame(data)

    with open(clustering_json, 'r') as f:
        cluster_labels = json.load(f)
    df['cluster_id'] = cluster_labels
    
    # Create a boolean column indicating whether the structure is a medoid of a cluster
    with open(medoid_indices_path, 'r') as f:
        medoid_idxs = json.load(f) # Load medoid indices
    df['is_medoid'] = df.index.isin(medoid_idxs)
    
    with open(embedding_2d_json, 'r') as f:
        embedding_xy = json.load(f)
    df['x'] = embedding_xy['x']
    df['y'] = embedding_xy['y']

    df.to_csv(result_csv_path, index_label='conf_id', sep=';')

def generate_cluster_dfs(point_df: pd.DataFrame, parameters: dict[str, str | int]) -> pd.DataFrame:
    # Create df of methods that found each cluster
    clusters = point_df['cluster_id'].unique().tolist()
    all_methods = point_df['method'].unique().tolist()
    cluster_found_data = {cluster: {} for cluster in clusters}
    for cur_cluster in clusters:
        methods_found = point_df[point_df['cluster_id'] == cur_cluster]['method'].unique().tolist()
        cluster_found_data[cur_cluster] = {
            f'{method}_found': method in methods_found
            for method in all_methods
        }
    found_df = pd.DataFrame.from_dict(cluster_found_data, orient='index')
    found_df.index.name = 'cluster_id'
    found_df = found_df.reset_index()

    # Assemble found-data and mean points in a single df
    assert parameters['expand_clusters']
    res_df = pd.merge(point_df, found_df, on='cluster_id')
    
    # # Highlight unassigned points
    #     res_df['unclustered'] = res_df['cluster'] == 0
    assert not parameters['separate_unclustered']
    res_df['unclustered'] = False
    
    if 'time' in res_df.columns:
        check_df = res_df.drop('time', axis=1)
    else:
        check_df = res_df

    assert not check_df.isna().any().any(), repr(res_df)
    return res_df

def get_relation(ringo_found, other_found, unclustered=None, methods=None, ref_method=None, base_method=None):
    # Check if both arrays have the same length
    if len(ringo_found) != len(other_found):
        raise ValueError("Arrays must have the same length.")
    
    # Create a new array to store the combined values
    combined_arr = np.empty(len(ringo_found), dtype=object)
    
    # Iterate over the arrays and assign 'both' to the corresponding element if both conditions are True
    for i in range(len(ringo_found)):
        if unclustered is not None and unclustered[i]:
            if methods[i] == base_method:
                combined_arr[i] = 'ringo_unclustered'
            elif methods[i] == ref_method:
                combined_arr[i] = 'other_unclustered'
            else:
                combined_arr[i] = None
        else:
            if ringo_found[i] and other_found[i]:
                combined_arr[i] = 'both'
            elif ringo_found[i]:
                combined_arr[i] = 'ringo'
            elif other_found[i]:
                combined_arr[i] = 'other'
            else:
                combined_arr[i] = None
    return combined_arr

def prep_timings_data(data_item, dfs_item, result_item):
    divplot_types = set(
        keys['divplot_type']
        for _, keys in dfs_item
    )

    for divplot_type in divplot_types:
        needed_combinations = {}
        for df, df_keys in dfs_item:
            if df_keys['divplot_type'] != divplot_type:
                continue
            method_to_timelimit = {}
            for cur_method in df['method'].unique():
                cur_timelimits = df[df['method'] == cur_method]['timelimit'].unique()
                assert len(cur_timelimits) == 1
                cur_timelimit = cur_timelimits[0]
                method_to_timelimit[cur_method] = cur_timelimit

            combo = df_keys['relativity'], df_keys['testset'], df_keys['testcase']
            needed_combinations[combo] = method_to_timelimit
        
        # merged_keys=['relativity', 'level', 'testset', 'testcase', 'method', 'timelimit', 'divplot_type'],
        # merged_keys=['divplot_type', 'method', 'timelimit'],
        for stats, stats_keys in data_item:
            combo = stats_keys['relativity'], stats_keys['testset'], stats_keys['testcase']
            if combo not in needed_combinations:
                continue
            method_to_timelimit = needed_combinations[combo]
            found_methods = []
            for method, timelimit in method_to_timelimit.items():
                if not (stats_keys['method'] == method and str(stats_keys['timelimit']) == str(timelimit)):
                    continue
                
                found_methods.append(method)
            if len(found_methods) == 0:
                continue
            result_item.include_element(
                stats,
                relativity=stats_keys['relativity'],
                level=stats_keys['level'],
                testset=stats_keys['testset'],
                testcase=stats_keys['testcase'],
                divplot_type=divplot_type,
                method=found_methods[0],
            )
            if len(found_methods) > 1:
                ic(found_methods)
                raise Exception('Fail')

def merge_timing_info(source_item, target_item):
    target_item.include_element({
        keys['method']: element
        for element, keys in source_item
    })

def generate_diversity_dfs(df: pd.DataFrame, parameters: dict[str, any], sampling_stats) -> pd.DataFrame:
    base_method_name: str = parameters['base_method']
    method_names: dict[str, str] = parameters['method_names']
    method_labels: dict[str, str] = parameters['method_labels']
    methodtype_ordering: list[str] = parameters['methodtype_ordering']

    other_methods = tuple(
        method
        for method in method_names.keys()
        if method != base_method_name and f'{method}_found' in df.columns
    )
    
    if f'{base_method_name}_found' not in df.columns:
        return None
    if len(other_methods) == 0:
        return None
    
    for method in other_methods:
        df[f'relation_{method}'] = get_relation(
            df[f'{base_method_name}_found'],
            df[f'{method}_found'],
            methods=df['method'],
            unclustered=df['unclustered'],
            ref_method=method,
            base_method=base_method_name
        )
    # ic(other_methods, df)
    # df.to_csv('check.csv', index=False, sep=';')
    
    # timings = get_method_timings(testcase)
    NAME_MAPS = {}
    def get_timing(method):
        return sampling_stats[method]['time']

    dfs: dict[str, pd.DataFrame] = {}
    for method in other_methods:
        remaining_methods = [m for m in other_methods if m != method]
        drop_axes = []
        # ic(method, remaining_methods)
        for m in remaining_methods:
            drop_axes.append(f'relation_{m}')
        drop_axes.append('time')
        #     drop_axes.append(f'{m}_found')
        # drop_axes.append(f'{base_method_name}_found')
        # drop_axes.append('cluster')
        # drop_axes.append(f'{method}_found')

        dfs[method] = (
            df.copy().drop(drop_axes, axis=1)
            .dropna()
            .rename(columns={f'relation_{method}': 'Cluster located by'})
            .replace(method_labels)
            .reset_index()
        )
        # dfs[method].to_csv(f'check_{method}.csv', index=False, sep=';')
        # Convert the column to a categorical type with the custom order
        dfs[method]['Cluster located by'] = dfs[method]['Cluster located by'].astype(
            pd.CategoricalDtype(categories=methodtype_ordering, ordered=True)
        )

        if parameters['show_times_slower']:
            times_speed = get_timing(method) / get_timing(base_method_name)
            if str(round(times_speed, 1)) != '1.0': 
                if times_speed > 1.0:
                    speed_word = 'slower'
                else:
                    times_speed = 1 / times_speed
                    speed_word = 'faster'
                speed_line = f'\n({times_speed:.1f}x {speed_word})'
            else:
                speed_line = ''
        elif parameters['show_exact_alttime']:
            speed_line = '\n' + str(datetime.timedelta(seconds=get_timing(method))).replace('day', 'd').split('.')[0]
        
        NAME_MAPS[method] = method_names[method] + speed_line
        dfs[method]['alt_method'] = NAME_MAPS[method]
        dfs[method] = dfs[method].sort_values(by='Cluster located by')
        # ic(dfs[method])

    plotdf = pd.concat([df for df in dfs.values()], ignore_index=True)
    return plotdf


def plot_confspace_maps():
    import sys
    import os
    module_dir = INSERT_HERE
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    os.chdir(os.path.dirname(__file__))

    import numpy as np
    import pandas as pd
    import rpy2.robjects as robjects
    import rpy2.rinterface as rinterface
    from rpy2.robjects import pandas2ri
    from scipy.spatial.distance import cdist
    
    from icecream import install
    install()

    diversity_df_csv = INSERT_HERE
    assert os.path.exists(diversity_df_csv)
    diversity_plot_png = INSERT_HERE

    mode_settings = INSERT_HERE
    method_labels = mode_settings['method_labels']
    methodtype_ordering = mode_settings['methodtype_ordering']

    # Restart the R session
    rinterface.initr()
    pandas2ri.activate()
    
    plotdf = pd.read_csv(diversity_df_csv, sep=';')

    # Find the range of 'x' and 'y' columns
    x_range = plotdf['x'].max() - plotdf['x'].min()
    y_range = plotdf['y'].max() - plotdf['y'].min()
    # Calculate scaling factors to fit the points in a unit square
    x_scale = 1 / x_range
    y_scale = 1 / y_range
    # Apply the scaling transformation to 'x' and 'y' columns
    plotdf['x'] = (plotdf['x'] - plotdf['x'].min()) * x_scale
    plotdf['y'] = (plotdf['y'] - plotdf['y'].min()) * y_scale

    # Pass plotdf into R
    num_of_methods = len(plotdf['alt_method'].unique())
    if num_of_methods > 2:
        plot_width = 12
    else:
        plot_width = 8

    r_plotdf = pandas2ri.py2rpy(plotdf)
    robjects.globalenv['plotdf'] = r_plotdf

    # Pass colors into R
    custom_colors = {
        method_labels['ringo']: '#4daf4a', # Green
        method_labels['other']: '#e41a1c', # Red
        method_labels['both']: '#377eb8', # Blue
        method_labels['ringo_unclustered']: '#777777', # Grey
        method_labels['other_unclustered']: '#000000', # Black
        method_labels['other_explored']: '#e6ab02', # Yellow
        # 'total_explored': '#ffffff', # white
    }
    r_custom_colors = robjects.vectors.ListVector(custom_colors)
    robjects.globalenv['custom_colors'] = r_custom_colors
    
    # Pass point sizes into R
    custom_sizes = {
        method_labels['ringo']: 0.4,
        method_labels['other']: 0.4,
        method_labels['both']: 0.4,
        method_labels['ringo_unclustered']: 0.3,
        method_labels['other_unclustered']: 0.3,
    }
    r_custom_sizes = robjects.vectors.ListVector(custom_sizes)
    robjects.globalenv['custom_sizes'] = r_custom_sizes
    
    # Pass the desired order of point layering
    r_custom_order = robjects.vectors.StrVector(list(reversed(methodtype_ordering)))
    robjects.globalenv['custom_order'] = r_custom_order

    MAIN_ISOVALUE = 0.02
    CURVE_COLORS = {
        'leveldf_altall': '#e6ab02',
        'leveldf_altexcl': '#ff7f00',
        'leveldf_mcrexcl': '#984ea3',
    }
    CURVE_DATA = {}
    if mode_settings['show_alternative_all']:
        # Pass the density df
        densitydf_total = plotdf.copy()
        densitydf_total['pointtype'] = 'total_explored'
        densitydf_alt = plotdf.copy()
        densitydf_alt = densitydf_alt[(densitydf_alt['Cluster located by'] != method_labels['ringo']) &\
                                    (densitydf_alt['Cluster located by'] != method_labels['ringo_unclustered'])]
        densitydf_alt['pointtype'] = method_labels['other_explored']
        densitydf  = densitydf_alt # pd.concat([], ignore_index=True)
        densitydf = densitydf[['x', 'y', 'alt_method']].reset_index(drop=True)
        
        methods = plotdf['alt_method'].unique()
        leveldfs: list[pd.DataFrame] = []
        for method in methods:
            x = np.linspace(0, 1, 100)
            y = np.linspace(0, 1, 100)
            # Create a meshgrid of x and y
            X, Y = np.meshgrid(x, y)
            # Calculate distances using cdist
            distances = cdist(densitydf[densitydf['alt_method'] == method][['x', 'y']], np.column_stack((X.flatten(), Y.flatten())))
            # Find the minimum distance for each point
            min_distances = np.min(distances, axis=0)
            # Create a DataFrame using Pandas
            data = {'x': X.flatten(), 'y': Y.flatten(), 'z': min_distances}
            newdf = pd.DataFrame(data)
            newdf['alt_method'] = method
            leveldfs.append(newdf)
        leveldf = pd.concat(leveldfs, ignore_index=True)
        leveldf['alt_method'] = leveldf['alt_method'].astype(plotdf['alt_method'].dtype)
        r_leveldf = pandas2ri.py2rpy(leveldf)
        curve_rname = 'leveldf_altall'
        robjects.globalenv[curve_rname] = r_leveldf
        CURVE_DATA[curve_rname] = CURVE_COLORS[curve_rname]

    if mode_settings['show_alternative_exclusive']:
        alt_densitydf = plotdf.copy()
        alt_densitydf = alt_densitydf[
            (alt_densitydf['Cluster located by'] != method_labels['ringo']) &\
            (alt_densitydf['Cluster located by'] != method_labels['ringo_unclustered'])
        ]
        alt_densitydf = alt_densitydf[['x', 'y', 'alt_method']].reset_index(drop=True)

        ringo_densitydf = plotdf.copy()
        ringo_densitydf = ringo_densitydf[
            (ringo_densitydf['Cluster located by'] == method_labels['ringo']) |\
            (ringo_densitydf['Cluster located by'] == method_labels['ringo_unclustered']) |\
            (ringo_densitydf['Cluster located by'] == method_labels['both'])
        ]
        ringo_densitydf = ringo_densitydf[['x', 'y', 'alt_method']].reset_index(drop=True)

        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        X, Y = np.meshgrid(x, y)
        ringo_distances = cdist(ringo_densitydf[['x', 'y']], np.column_stack((X.flatten(), Y.flatten())))
        min_ringo_distances = np.min(ringo_distances, axis=0)
        
        leveldfs = []
        methods = plotdf['alt_method'].unique()
        for method in methods:
            alt_distances = cdist(alt_densitydf[alt_densitydf['alt_method'] == method][['x', 'y']], np.column_stack((X.flatten(), Y.flatten())))
            min_alt_distances = np.min(alt_distances, axis=0)
            assert len(min_alt_distances) == len(min_ringo_distances)
            for i, (ringo, alt) in enumerate(zip(min_ringo_distances, min_alt_distances)):
                if ringo < MAIN_ISOVALUE:
                    min_alt_distances[i] = 1.0
            # Create a DataFrame using Pandas
            data = {'x': X.flatten(), 'y': Y.flatten(), 'z': min_alt_distances}
            newdf = pd.DataFrame(data)
            newdf['alt_method'] = method
            leveldfs.append(newdf)
        leveldf = pd.concat(leveldfs, ignore_index=True)
        
        r_leveldf = pandas2ri.py2rpy(leveldf)
        curve_rname = 'leveldf_altexcl'
        robjects.globalenv[curve_rname] = r_leveldf
        CURVE_DATA[curve_rname] = CURVE_COLORS[curve_rname]

    if mode_settings['show_mcr_exclusive']:
        alt_densitydf = plotdf.copy()
        alt_densitydf = alt_densitydf[
            (alt_densitydf['Cluster located by'] != method_labels['ringo']) &\
            (alt_densitydf['Cluster located by'] != method_labels['ringo_unclustered'])
        ]
        alt_densitydf = alt_densitydf[['x', 'y', 'alt_method']].reset_index(drop=True)

        ringo_densitydf = plotdf.copy()
        ringo_densitydf = ringo_densitydf[
            (ringo_densitydf['Cluster located by'] == method_labels['ringo']) |\
            (ringo_densitydf['Cluster located by'] == method_labels['ringo_unclustered']) |\
            (ringo_densitydf['Cluster located by'] == method_labels['both'])
        ]
        ringo_densitydf = ringo_densitydf[['x', 'y', 'alt_method']].reset_index(drop=True)

        
        leveldfs = []
        methods = plotdf['alt_method'].unique()
        for method in methods:
            x = np.linspace(0, 1, 100)
            y = np.linspace(0, 1, 100)
            X, Y = np.meshgrid(x, y)
            ringo_distances = cdist(ringo_densitydf[['x', 'y']], np.column_stack((X.flatten(), Y.flatten())))
            min_ringo_distances = np.min(ringo_distances, axis=0)
            
            alt_distances = cdist(alt_densitydf[alt_densitydf['alt_method'] == method][['x', 'y']], np.column_stack((X.flatten(), Y.flatten())))
            min_alt_distances = np.min(alt_distances, axis=0)
            assert len(min_alt_distances) == len(min_ringo_distances)
            for i, (ringo, alt) in enumerate(zip(min_ringo_distances, min_alt_distances)):
                if alt < MAIN_ISOVALUE:
                    min_ringo_distances[i] = 1.0
            # Create a DataFrame using Pandas
            data = {'x': X.flatten(), 'y': Y.flatten(), 'z': min_ringo_distances}
            newdf = pd.DataFrame(data)
            newdf['alt_method'] = method
            leveldfs.append(newdf)
        leveldf = pd.concat(leveldfs, ignore_index=True)
        
        r_leveldf = pandas2ri.py2rpy(leveldf)        
        curve_rname = 'leveldf_mcrexcl'
        robjects.globalenv[curve_rname] = r_leveldf
        CURVE_DATA[curve_rname] = CURVE_COLORS[curve_rname]

    contour_isovalues = [MAIN_ISOVALUE,]
    # Pass the desired order of point layering
    r_contour_isovalues = robjects.vectors.FloatVector(list(reversed(contour_isovalues)))
    robjects.globalenv['contour_isovalues'] = r_contour_isovalues
    
    # r_method_ordering = robjects.vectors.StrVector(list(plotdf['alt_method'].cat.categories))
    # robjects.globalenv['method_ordering'] = r_method_ordering
    
    def plot(safe=False):
        # Execute R code
        optional_density = ""
        if not safe:
            for dfname, color in CURVE_DATA.items():
                optional_density += f"geom_contour({dfname}, mapping = aes(x = x, y = y, z = z), colour='{color}', breaks = contour_isovalues) +"
            # optional_density = f"geom_tile(leveldf, mapping = aes(x = x, y = y, fill = z)) +"
        # if res_fname.endswith('.svg'):
        #     svg_addition = ", device = svg"
        # else:
        #     svg_addition = ""
        svg_addition = ""
        # plotdf$alt_method_f <- factor(plotdf$alt_method, levels = method_ordering, ordered = TRUE)
        # plotdf$alt_method_f <- factor(plotdf$alt_method, levels = method_ordering, ordered = TRUE)
        # leveldf$alt_method_f <- factor(leveldf$alt_method, levels = method_ordering, ordered = TRUE)
        rcode = f"""\
    library(ggplot2)

    facet_theme <- theme_bw() +
        theme(panel.grid.major = element_blank(),
            panel.grid.minor = element_blank(),
            panel.border = element_rect(colour = "black", fill = NA, linewidth = 1),
            axis.line = element_line(colour = "black"),
            axis.title = element_text(size = 12, face = "bold"),
            axis.text = element_text(size = 14),
            legend.title = element_text(size = 14, face = "bold"),
            legend.text = element_text(size = 14),
            strip.text = element_text(size = 14, face = "bold")
        )
    remove_axes <- theme(
        axis.text.x = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks = element_blank(),
        axis.line = element_blank(),
        axis.title.y = element_blank()
    )

    # Create the plot
    custom_sizes_values <- unlist(custom_sizes)
    # Reorder the rows in the dataframe based on the desired order
    plotdf$`Cluster located by` <- factor(plotdf$`Cluster located by`, levels = custom_order)
    plotdf <- plotdf[order(plotdf$`Cluster located by`), ]

    plot <- ggplot() +
        geom_point(plotdf, mapping = aes(x = x, y = y, color = `Cluster located by`, size = `Cluster located by`)) +
        scale_size_manual(values = custom_sizes_values) + {optional_density}
        facet_grid(. ~ alt_method) +
        facet_theme + remove_axes +
        scale_color_manual(values = custom_colors) +
        labs(x = '', title='Diversity benchmark') +
        theme(text=element_text(family="Arial"))
    ggsave(plot, filename = "{diversity_plot_png}", width = {plot_width}, height = 3 {svg_addition})
    """
        robjects.r(rcode)

    try:
        plot()
    except:
        plot(safe=True)


def build_diversity_summary(png_item, pdf_item, mode_settings) -> None:
    from chemscripts.imageutils import HImageLayout, VImageLayout, pil_to_reportlab
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from PIL import Image

    TITLE_SIZE = 48

    pdf_path: str = pdf_item.get_path()
    c = canvas.Canvas(pdf_path, pagesize=letter)
    all_testcases = set(
        keys['testcase']
        for _, keys in png_item
    )
    all_testcases = sorted(list(all_testcases))
    heavy_testcases = ['pdb2IYA', 'csdMIWTER', 'csdYIVNOG', 'pdb2C6H', 'pdb3M6G', 'pdb2QZK', 'pdb1NWX', 'csdRECRAT', 'csdFINWEE10', 'csdRULSUN']
    easy_testcases = [
        testcase
        for testcase in all_testcases
        if testcase not in heavy_testcases
    ]

    for testcase in heavy_testcases + easy_testcases:
        if testcase not in all_testcases:
            continue
        cur_plot = None
        for png_name, keys in png_item:
            if keys['testcase'] == testcase:
                cur_plot = Image.open(png_name)
                cur_plot = cur_plot.resize((int(cur_plot.width/2), int(cur_plot.height/2)), Image.LANCZOS)
        assert cur_plot is not None

        c.setPageSize((cur_plot.size[0], cur_plot.size[1] + TITLE_SIZE))
        c.drawImage(pil_to_reportlab(cur_plot), 0, 0, width=cur_plot.width, height=cur_plot.height)
        c.setFont("Helvetica", 32)
        c.drawString(
            0, cur_plot.size[1],
            f"{confsearch.format_testcase(testcase)}"
        )
        c.showPage()
    
    c.save()
    pdf_item.include_element(pdf_path)


def extract_data_for_crosscomparison(csvs, base_ensembles, method_ensembles, extracted_data):
    accept_conditions = {
        'relativity': 'basic',
        'level': 'mmff',
        'divplot_type': 'basicOFive',
        'testset': 'macromodel',
    }

    for diversity_csv, csv_keys in csvs:
        if not all(
            csv_keys[k] == v
            for k, v in accept_conditions.items()
        ):
            ic('Skip1', diversity_csv)
            continue
        df = pd.read_csv(diversity_csv, sep=';')
        covered_methods = df['method'].unique()
        ic(covered_methods)
        if 'crestOld' not in covered_methods:# or 'ringo' not in covered_methods:
            ic('Skip2', diversity_csv)
            continue
        ic('Accepted', diversity_csv)

        base_xyz = base_ensembles.access_element(
            relativity=csv_keys['relativity'],
            level=csv_keys['level'],
            divplot_type=csv_keys['divplot_type'],
            testset=csv_keys['testset'],
            testcase=csv_keys['testcase'],
        )
        long_xyz = method_ensembles.access_element(
            opttype='mmff',
            testset=csv_keys['testset'],
            testcase=csv_keys['testcase'],
            method='ringo-vs-crest',
            timelimit='long',
        )
        result_element = {
            'csv': diversity_csv,
            'base_xyz': base_xyz,
            'long_xyz': long_xyz,
        }
        extracted_data.include_element(
            result_element,
            relativity=csv_keys['relativity'],
            level=csv_keys['level'],
            divplot_type=csv_keys['divplot_type'],
            testset=csv_keys['testset'],
            testcase=csv_keys['testcase'],
        )
        ic('included', result_element)


def run_vscrest_comparison(diversity_crosscompare_data):
    import sys
    import os
    module_dir = INSERT_HERE
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    os.chdir(os.path.dirname(__file__))

    import json
    import ringo
    from utils.confsearch import ConformerInfo
    import networkx as nx
    import pandas as pd

    from icecream import install
    install()

    diversity_crosscompare_data = INSERT_HERE
    result_json_path = INSERT_HERE

    csv_path = diversity_crosscompare_data['csv']
    base_ensemble_path = diversity_crosscompare_data['base_xyz']
    long_ensemble_path = diversity_crosscompare_data['long_xyz']

    diversity_df = pd.read_csv(csv_path, sep=';')
    # diversity_df = diversity_df[(diversity_df['Cluster located by'] == 'Alternative (only)') ]
    diversity_df = diversity_df[(diversity_df['Cluster located by'] == 'Alternative (only)') & (diversity_df['alt_method'].str.contains('CREST'))]
    missed_conformer_names = diversity_df['name'].tolist()
    
    p = ringo.Confpool()
    p.include_from_file(base_ensemble_path)
    def base_filter(m) -> bool:
        data = ConformerInfo(description=m.descr).data
        return (
            data['generated']['method'] == 'crestOld' and
            data['name'] in missed_conformer_names
        )
    p.filter(base_filter)
    ic([m.descr for m in p])

    first_check_index = len(p)
    max_missed_index = len(p) - 1
    p.include_from_file(long_ensemble_path)
    
    p.generate_connectivity(0, mult=1.3)
    graph = p.get_connectivity().copy()
    all_nodes = [i for i in graph.nodes]
    bridges = list(nx.bridges(graph))
    graph.remove_edges_from(bridges)
    # Some of these connected components will be out cyclic parts, others are just single atoms
    components_lists = [list(comp) for comp in nx.connected_components(graph)]

    # Compute separate RMSD matrices with respect to each cyclic part
    crmsd_list = []
    for check_conf_index in range(max_missed_index + 1):
        check_conformer_name = ConformerInfo(description=p[check_conf_index].descr).data['name']
        ic(check_conf_index, max_missed_index)
        rmsd_values = []
        for conn_component in components_lists:
            if len(conn_component) == 1:
                continue

            p.generate_connectivity(0, mult=1.3,
                                    ignore_elements=[node for node in all_nodes 
                                                    if node not in conn_component])
            cur_graph = p.get_connectivity()
            assert cur_graph.number_of_nodes() == len(conn_component)
            p.generate_isomorphisms()
            cur_rmsd_values = []
            for j in range(first_check_index, len(p)):
                rmsd_value, _, __ = p[check_conf_index].rmsd(p[j])
                cur_rmsd_values.append(rmsd_value)
            rmsd_values.append(cur_rmsd_values)
        final_rmsd_values = [
            (i, max((
                rmsd_values[j][i]
                for j in range(len(rmsd_values))
            )))
            for i in range(len(rmsd_values[0]))
        ]
        best_index, best_crmsd = min(final_rmsd_values, key=lambda x: x[1])
        best_name = ConformerInfo(description=p[best_index + first_check_index].descr).data['name']
        crmsd_list.append((check_conformer_name, best_name, best_crmsd))
    ic(crmsd_list)

    with open(result_json_path, 'w') as f:
        json.dump(crmsd_list ,f)


def vscrest_analyze(data, input_files):
    import ringo
    from utils.confsearch import ConformerInfo

    df = {
        'check_conformer_name': [],
        'closest_conformer_name': [],
        'min_crmsd': [],
        'check_energy': [],
    }

    # long_ensemble_path = input_files['long_xyz']
    # long_p = ringo.Confpool()
    base_ensemble_path = input_files['base_xyz']
    base_p = ringo.Confpool()
    base_p.include_from_file(base_ensemble_path)
    base_p.filter(lambda m: 'crestOld' in m.descr)
    def get_energy(confname):
        result = None
        for m in base_p:
            data = ConformerInfo(description=m.descr).data
            if confname == data['name']:
                result = data['relenergy']
                break
        assert result is not None, confname
        return result

    for row in data:
        df['check_conformer_name'].append(row[0])
        df['closest_conformer_name'].append(row[1])
        df['min_crmsd'].append(row[2])
        df['check_energy'].append(get_energy(row[0]))
    df = pd.DataFrame(df)
    df = df.sort_values(by=['min_crmsd'], ascending=False).reset_index(drop=True)
    return df


def vscrest_summarize(raw_dfs_item, res_item):
    summary_df = {
        'testset': [],
        'testcase': [],
        'crmsd_interval': [],
        'num_missed_conformers': [],
        'min_check_energy': [],
    }
    for raw_df, keys in raw_dfs_item:
        max_crmsd_difference = raw_df['min_crmsd'].max()
        crmsd_segments = {}
        i = 0.0
        while i < max_crmsd_difference:
            min_max = (i, i + 0.1)
            crmsd_segments[min_max] = f"{min_max[0]:.1f}-{min_max[1]:.1f}"
            i += 0.1
        
        testset = keys['testset']
        testcase = keys['testcase']

        raw_df = raw_df.to_dict(orient='records')
        segment_counts = {
            name: 0
            for name in crmsd_segments.values()
        }
        segment_minenergy = {
            name: None
            for name in crmsd_segments.values()
        }
        for row in raw_df:
            interval_name = None
            for seg, seg_name in crmsd_segments.items():
                if row['min_crmsd'] > seg[0] and row['min_crmsd'] < seg[1]:
                    interval_name = seg_name
                    break
            assert interval_name is not None
            segment_counts[interval_name] += 1
            cur_energy = row['check_energy']
            if segment_minenergy[interval_name] is None or cur_energy < segment_minenergy[interval_name]:
                segment_minenergy[interval_name] = cur_energy
        
        for seg_name, seg_count in segment_counts.items():
            if seg_count == 0:
                continue
            summary_df['testset'].append(testset)
            summary_df['testcase'].append(testcase)
            summary_df['crmsd_interval'].append(seg_name)
            summary_df['num_missed_conformers'].append(seg_count)
            summary_df['min_check_energy'].append(segment_minenergy[seg_name])

    res_item.include_element(pd.DataFrame(summary_df))
        


def diversity_transforms(ds, main_logger, maxproc: int=1) -> list[Transform]:
    
    diversity_transforms: list[Transform] = [
        templates.map('load_diversity_ensemble_xyz_paths',
            input='final_ensemble_path', output='diversity_ensemble_xyz_paths',
            aware_keys=['relativity', 'level', 'method', 'timelimit'],
            mapping=lambda final_ensemble_path, **kw:
                load_diversity_ensemble_xyz_paths(xyz_path=final_ensemble_path, **kw)
        ),
        templates.exec('prepare_ensemble_relenergies_for_diversity',
            input=['ensemble_relenergies_json', 'diversity_ensemble_xyz_paths'],
            output='ensemble_relenergies_json_for_diversity',
            merged_keys=['relativity', 'level', 'testset', 'testcase', 'method', 'timelimit', 'divplot_type'],
            method=lambda ensemble_relenergies_json, diversity_ensemble_xyz_paths, ensemble_relenergies_json_for_diversity:
                select_clustered_elements(
                    all_elements=ensemble_relenergies_json,
                    ref_item=diversity_ensemble_xyz_paths,
                    selected_elements=ensemble_relenergies_json_for_diversity,
                )
        ),
        templates.pyfunction_subprocess('prepare_initial_ensembles',
            input=['diversity_ensemble_xyz_paths','ensemble_relenergies_json_for_diversity'],
            output='initial_diversity_ensemble', aware_keys=['divplot_type'],
            pyfunction=prepare_initial_ensembles, calcdir='calcdir', nproc=1,
            argv_prepare=lambda diversity_ensemble_xyz_paths, ensemble_relenergies_json_for_diversity, initial_diversity_ensemble, **kw: (
                diversity_ensemble_xyz_paths.access_element(), ensemble_relenergies_json_for_diversity.access_element(), initial_diversity_ensemble.get_path()
            ),
            subs=lambda diversity_ensemble_xyz_paths, ensemble_relenergies_json_for_diversity, initial_diversity_ensemble, divplot_type, **kw: {
                'raw_xyz_path': diversity_ensemble_xyz_paths.access_element(),
                'relenergies_json': ensemble_relenergies_json_for_diversity.access_element(),
                'target_xyz_path': initial_diversity_ensemble.get_path(),
                'mode_settings': DIVERSITY_ANALYSIS_MODES[divplot_type]['parameters'],
                'CRMSD_PYXYZ_SETTINGS': CRMSD_PYXYZ_SETTINGS,
            },
            output_process=lambda initial_diversity_ensemble, **kw: confsearch.load_if_exists(initial_diversity_ensemble)
        ),
        templates.pyfunction_subprocess('merge_initial_ensembles',
            input='initial_diversity_ensemble', output='merged_diversity_ensemble',
            aware_keys=['testset', 'testcase', 'divplot_type'], merged_keys=['method', 'timelimit'],
            pyfunction=merge_initial_ensembles, calcdir='calcdir', nproc=1,
            argv_prepare=lambda testset, testcase, merged_diversity_ensemble, **kw:
                (testset, testcase, merged_diversity_ensemble.get_path()),
            subs=lambda initial_diversity_ensemble, merged_diversity_ensemble, divplot_type, **kw: {
                'input_xyz_paths': [(xyz_name, keys) for xyz_name, keys in initial_diversity_ensemble],
                'target_xyz_path': merged_diversity_ensemble.get_path(),
                'mode_settings': DIVERSITY_ANALYSIS_MODES[divplot_type]['parameters'],
            },
            output_process=lambda merged_diversity_ensemble, **kw: confsearch.assertive_include(merged_diversity_ensemble)
        ),
        templates.pyfunction_subprocess('generate_crmsd_matrix',
            input='merged_diversity_ensemble',
            output=['crmsd_matrix', 'final_diversity_ensemble'], aware_keys=['divplot_type'],
            pyfunction=generate_crmsd_matrix, calcdir='calcdir', nproc=1,
            argv_prepare=lambda merged_diversity_ensemble, crmsd_matrix, final_diversity_ensemble, **kw:
                (merged_diversity_ensemble.access_element(), crmsd_matrix.get_path(), final_diversity_ensemble.get_path()),
            subs=lambda merged_diversity_ensemble, crmsd_matrix, final_diversity_ensemble, divplot_type, **kw: {
                'xyz_path': merged_diversity_ensemble.access_element(),
                'crmsd_matrix_path': crmsd_matrix.get_path(),
                'target_xyz_path': final_diversity_ensemble.get_path(),
                'mode_settings': DIVERSITY_ANALYSIS_MODES[divplot_type]['parameters'],
                'CRMSD_PYXYZ_SETTINGS': CRMSD_PYXYZ_SETTINGS,
            },
            output_process=lambda crmsd_matrix, final_diversity_ensemble, **kw:
                (confsearch.load_if_exists(crmsd_matrix), confsearch.load_if_exists(final_diversity_ensemble))
        ).greedy_on('final_diversity_ensemble'),
        templates.pyfunction_subprocess('generate_all_clusterings',
            input='crmsd_matrix', output=['clustering_indices_json', 'clustering_reachability_plot'], aware_keys=['divplot_type'],
            pyfunction=generate_all_clusterings, calcdir='calcdir', nproc=1,
            argv_prepare=lambda crmsd_matrix, clustering_indices_json, clustering_reachability_plot, **kw:
                (crmsd_matrix.access_element(), clustering_indices_json.get_path(), clustering_reachability_plot.get_path()),
            subs=lambda crmsd_matrix, clustering_indices_json, clustering_reachability_plot, divplot_type, **kw: {
                'crmsd_matrix_path': crmsd_matrix.access_element(),
                'target_json_path': clustering_indices_json.get_path(),
                'target_png_path': clustering_reachability_plot.get_path(),
                'mode_settings': DIVERSITY_ANALYSIS_MODES[divplot_type]['parameters'],
            },
            output_process=lambda clustering_indices_json, clustering_reachability_plot, **kw:
                (confsearch.load_if_exists(clustering_indices_json), confsearch.load_if_exists(clustering_reachability_plot)),
            env_activation=RPYTHON_ACTIVATION
        ),
        templates.pyfunction_subprocess('get_best_clusterings',
            input=['clustering_indices_json', 'clustering_reachability_plot'],
            output=['bestclustering_indices_json', 'bestclustering_reachability_plot'],
            aware_keys=['testset', 'testcase', 'divplot_type'],
            pyfunction=find_best_clusterings, calcdir='calcdir', nproc=1,
            argv_prepare=lambda divplot_type, testset, testcase, **kw: (divplot_type, testset, testcase),
            subs=lambda clustering_indices_json, clustering_reachability_plot, bestclustering_indices_json, bestclustering_reachability_plot, divplot_type, **kw: {
                'input_clustering_json': clustering_indices_json.access_element(),
                'input_clustering_pngs_zip': clustering_reachability_plot.access_element(),
                'target_clustering_json': bestclustering_indices_json.get_path(),
                'target_clustering_png': bestclustering_reachability_plot.get_path(),
                'mode_settings': DIVERSITY_ANALYSIS_MODES[divplot_type]['parameters'],
            },
            output_process=lambda bestclustering_indices_json, bestclustering_reachability_plot, **kw:
                (confsearch.load_if_exists(bestclustering_indices_json), confsearch.load_if_exists(bestclustering_reachability_plot)),
        ),
        templates.exec('select_clustered_crmsd_matrices',
            input=['crmsd_matrix', 'bestclustering_indices_json'], output='clustered_crmsd_matrix_paths',
            merged_keys=['testset', 'testcase'],
            method=lambda crmsd_matrix, bestclustering_indices_json, clustered_crmsd_matrix_paths:
                select_clustered_elements(all_elements=crmsd_matrix, ref_item=bestclustering_indices_json, selected_elements=clustered_crmsd_matrix_paths)
        ),
        templates.pyfunction_subprocess('generate_cluster_medoids',
            input=['clustered_crmsd_matrix_paths', 'bestclustering_indices_json'], output='cluster_medoid_indices_json',
            aware_keys=['testset', 'testcase', 'divplot_type'],
            pyfunction=generate_cluster_medoids, calcdir='calcdir', nproc=1,
            argv_prepare=lambda divplot_type, testset, testcase, **kw: (divplot_type, testset, testcase),
            subs=lambda clustered_crmsd_matrix_paths, bestclustering_indices_json, cluster_medoid_indices_json, divplot_type, **kw: {
                'crmsd_matrix_path': clustered_crmsd_matrix_paths.access_element(),
                'bestclustering_indices_json': bestclustering_indices_json.access_element(),
                'cluster_medoid_indices_json': cluster_medoid_indices_json.get_path(),
                'mode_settings': DIVERSITY_ANALYSIS_MODES[divplot_type]['parameters'],
            },
            output_process=lambda cluster_medoid_indices_json, **kw: confsearch.assertive_include(cluster_medoid_indices_json),
        ),
        templates.pyfunction_subprocess('generate_2d_embeddings',
            input='clustered_crmsd_matrix_paths', output='embedding_2d_json',
            aware_keys=['testset', 'testcase', 'divplot_type'],
            pyfunction=generate_2d_embeddings, calcdir='calcdir', nproc=int(maxproc/2),
            argv_prepare=lambda clustered_crmsd_matrix_paths, **kw: (clustered_crmsd_matrix_paths.access_element(),),
            subs=lambda clustered_crmsd_matrix_paths, embedding_2d_json, divplot_type, **kw: {
                'crmsd_matrix_path': clustered_crmsd_matrix_paths.access_element(),
                'embedding_2d_json_path': embedding_2d_json.get_path(),
                'mode_settings': DIVERSITY_ANALYSIS_MODES[divplot_type]['parameters'],
            },
            output_process=lambda embedding_2d_json, **kw: confsearch.assertive_include(embedding_2d_json),
        ).greedy_on('embedding_2d_json'),
        templates.exec('select_clustered_final_ensembles',
            input=['final_diversity_ensemble', 'bestclustering_indices_json'], output='clustered_final_ensemble_paths',
            merged_keys=['testset', 'testcase'],
            method=lambda final_diversity_ensemble, bestclustering_indices_json, clustered_final_ensemble_paths:
                select_clustered_elements(all_elements=final_diversity_ensemble, ref_item=bestclustering_indices_json, selected_elements=clustered_final_ensemble_paths)
        ),
        templates.pyfunction_subprocess('generate_point_dfs',
            input=['clustered_final_ensemble_paths', 'bestclustering_indices_json', 'cluster_medoid_indices_json', 'embedding_2d_json'],
            output='point_df_csvs', aware_keys=['testset', 'testcase', 'divplot_type'],
            pyfunction=generate_point_dfs, calcdir='calcdir', nproc=1,
            argv_prepare=lambda clustered_final_ensemble_paths, **kw: (clustered_final_ensemble_paths.access_element(),),
            subs=lambda clustered_final_ensemble_paths, bestclustering_indices_json, cluster_medoid_indices_json, embedding_2d_json, point_df_csvs, divplot_type, **kw: {
                'ensemble_xyz_path': clustered_final_ensemble_paths.access_element(),
                'clustering_json': bestclustering_indices_json.access_element(),
                'medoid_indices_path': cluster_medoid_indices_json.access_element(),
                'embedding_2d_json': embedding_2d_json.access_element(),
                'result_csv_path': point_df_csvs.get_path(),
                'mode_settings': DIVERSITY_ANALYSIS_MODES[divplot_type]['parameters'],
            },
            output_process=lambda point_df_csvs, **kw: confsearch.assertive_include(point_df_csvs),
        ),
        templates.pd_from_csv('load_point_dfs',
            input='point_df_csvs', output='point_dfs', sep=';'
        ),
        templates.map('generate_cluster_dfs',
            input='point_dfs', output='cluster_dfs', aware_keys=['divplot_type'],
            mapping=lambda point_dfs, divplot_type: generate_cluster_dfs(
                point_df=point_dfs,
                parameters=DIVERSITY_ANALYSIS_MODES[divplot_type]['parameters']
            ),
        ),
        templates.pd_to_csv('save_cluster_dfs',
            input='cluster_dfs', output='cluster_df_csvs', sep=';', index=False
        ),
        templates.pd_from_csv('load_cluster_dfs',
            input='cluster_df_csvs', output='cluster_dfs', sep=';'
        ),
        templates.exec('prep_timings_data_for_diversity',
            input=['total_single_csdata_obj', 'cluster_dfs'], output='timing_info_raw',
            # merged_keys=['divplot_type', 'method', 'timelimit'],
            merged_keys=['relativity', 'level', 'testset', 'testcase', 'method', 'timelimit', 'divplot_type'],
            method=lambda total_single_csdata_obj, cluster_dfs, timing_info_raw: prep_timings_data(
                data_item=total_single_csdata_obj,
                dfs_item=cluster_dfs,
                result_item=timing_info_raw,
            )
        ),
        templates.exec('merge_timing_info_diversity',
            input='timing_info_raw', output='timing_info', merged_keys=['method'],
            method=lambda timing_info_raw, timing_info:
                merge_timing_info(
                    source_item=timing_info_raw,
                    target_item=timing_info
                )
        ),
        templates.map('generate_diversity_dfs',
            input=['cluster_dfs', 'timing_info'], output='diversity_dfs', aware_keys=['divplot_type'],
            mapping=lambda cluster_dfs, divplot_type, timing_info: generate_diversity_dfs(
                df=cluster_dfs.copy(),
                parameters=DIVERSITY_ANALYSIS_MODES[divplot_type]['parameters'],
                sampling_stats=timing_info
            ),
            include_none=False
        ),
        templates.pd_to_csv('save_diversity_dfs',
            input='diversity_dfs', output='diversity_df_csvs', sep=';', index=False
        ),
        templates.pyfunction_subprocess('plot_confspace_maps_png',
            input='diversity_df_csvs', output='diversity_plot_png', aware_keys=['divplot_type'],
            pyfunction=plot_confspace_maps, calcdir='calcdir', nproc=4,
            argv_prepare=lambda diversity_df_csvs, diversity_plot_png, **kw:
                (diversity_df_csvs.access_element(), diversity_plot_png.get_path()),
            subs=lambda diversity_df_csvs, diversity_plot_png, divplot_type, **kw: {
                'diversity_df_csv': diversity_df_csvs.access_element(),
                'diversity_plot_png': diversity_plot_png.get_path(),
                'mode_settings': DIVERSITY_ANALYSIS_MODES[divplot_type]['parameters'],
            },
            output_process=lambda diversity_plot_png, **kw: confsearch.assertive_include(diversity_plot_png),
            env_activation=RPYTHON_ACTIVATION
        ),
        templates.pyfunction_subprocess('plot_confspace_maps_svg',
            input='diversity_df_csvs', output='diversity_plot_svg', aware_keys=['divplot_type'],
            pyfunction=plot_confspace_maps, calcdir='calcdir', nproc=4,
            argv_prepare=lambda diversity_df_csvs, diversity_plot_svg, **kw:
                (diversity_df_csvs.access_element(), diversity_plot_svg.get_path()),
            subs=lambda diversity_df_csvs, diversity_plot_svg, divplot_type, **kw: {
                'diversity_df_csv': diversity_df_csvs.access_element(),
                'diversity_plot_png': diversity_plot_svg.get_path(),
                'mode_settings': DIVERSITY_ANALYSIS_MODES[divplot_type]['parameters'],
            },
            output_process=lambda diversity_plot_svg, **kw: confsearch.assertive_include(diversity_plot_svg),
            env_activation=RPYTHON_ACTIVATION
        ),
        templates.exec('build_diversity_summary',
            input='diversity_plot_png', output='diversity_summary_pdf',
            aware_keys=['divplot_type'], merged_keys=['testset', 'testcase'],
            method=lambda diversity_plot_png, diversity_summary_pdf, divplot_type, **kw:
                build_diversity_summary(
                    png_item=diversity_plot_png,
                    pdf_item=diversity_summary_pdf,
                    mode_settings=DIVERSITY_ANALYSIS_MODES[divplot_type]['parameters']
                )
        ),

        # Verification that all clusters are discovered by MCR with CREST runtimes
        templates.exec('extract_vscrest_diversity_comparison',
            input=['diversity_df_csvs', 'final_diversity_ensemble', 'filtered_ensemble_xyz'], output='diversity_crosscompare_data',
            merged_keys=['relativity', 'level', 'testset', 'testcase', 'method', 'timelimit', 'divplot_type', 'opttype'],
            method=lambda diversity_df_csvs, final_diversity_ensemble, filtered_ensemble_xyz, diversity_crosscompare_data, **kw:
                extract_data_for_crosscomparison(
                    csvs=diversity_df_csvs,
                    base_ensembles=final_diversity_ensemble,
                    method_ensembles=filtered_ensemble_xyz,
                    extracted_data=diversity_crosscompare_data,
                )
        ),
        templates.load_json('load_vscrest_diversity_comparison',
            input='diversity_crosscompare_json', output='diversity_crosscompare_data'
        ),
        templates.dump_json('dump_vscrest_diversity_comparison',
            input='diversity_crosscompare_data', output='diversity_crosscompare_json'
        ),
        templates.pyfunction_subprocess('run_vscrest_diversity_comparison',
            input='diversity_crosscompare_data', output='diversity_crosscompare_results_json',
            pyfunction=run_vscrest_comparison, calcdir='calcdir', nproc=1,
            argv_prepare=lambda diversity_crosscompare_data, **kw: (diversity_crosscompare_data.access_element(),),
            subs=lambda diversity_crosscompare_data, diversity_crosscompare_results_json, **kw: {
                'diversity_crosscompare_data': diversity_crosscompare_data.access_element(),
                'result_json_path': diversity_crosscompare_results_json.get_path(),
            },
            output_process=lambda diversity_crosscompare_results_json, **kw: confsearch.assertive_include(diversity_crosscompare_results_json),
        ),
        templates.load_json('load_vscrest_diversity_comparison_json',
            input='diversity_crosscompare_results_json', output='diversity_crosscompare_results'
        ),
        templates.map('analyze_vscrest_diversity_comparison',
            input=['diversity_crosscompare_results', 'diversity_crosscompare_data'], output='diversity_crosscompare_analysis',
            mapping=lambda diversity_crosscompare_results, diversity_crosscompare_data:
                vscrest_analyze(data=diversity_crosscompare_results, input_files=diversity_crosscompare_data)
        ),
        templates.pd_to_csv('save_vscrest_diversity_comparison',
            input='diversity_crosscompare_analysis', output='diversity_crosscompare_analysis_csv', sep=';', index=False
        ),
        templates.pd_from_csv('load_vscrest_diversity_comparison_analysis',
            input='diversity_crosscompare_analysis_csv', output='diversity_crosscompare_analysis', sep=';'
        ),
        templates.exec('summarize_vscrest_diversity_comparison',
            input='diversity_crosscompare_analysis', output='diversity_crosscompare_summary',
            merged_keys=['testset', 'testcase'],
            method=lambda diversity_crosscompare_analysis, diversity_crosscompare_summary:
                vscrest_summarize(raw_dfs_item=diversity_crosscompare_analysis, res_item=diversity_crosscompare_summary)
        ),
        templates.pd_to_csv('save_vscrest_diversity_summary',
            input='diversity_crosscompare_summary', output='diversity_crosscompare_summary_csv', sep=';', index=False
        ),
    ]

    return diversity_transforms
