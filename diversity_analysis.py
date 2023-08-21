import glob, os, sys, json, shutil, time, random
from environments import PathHandler, parallelize_call
try:
    import numpy as np
    import pandas as pd
    import networkx as nx
except:
    # the main python interpreter is allowed to miss some of the modules
    assert __name__ == "__main__"

# Independent paths
MAIN_DF_NAME = 'benchmark_stats.csv'
START_XYZ_DIR = 'lower_conformers'
STARTING_RMSD_MATRIX_DIR = './complete_ensemble_diversity/rmsd_matrices'
STARTING_RMSD_MATRIX_FILE = os.path.join(STARTING_RMSD_MATRIX_DIR, '{molname}_rmsd.npz') # To decide which conformers are perspective

# Variable paths
PATHS = PathHandler()
PATHS['TOTALENSEMBLES_DIR'] = 'total_confensembles'
PATHS['TOTALENSEMBLE_XYZ'] = os.path.join(PATHS['TOTALENSEMBLES_DIR'], '{molname}_total.xyz')
PATHS['RMSDMATRIX_DIR'] = 'rmsd_matrices'
PATHS['RMSDMATRIX_FILE'] = os.path.join(PATHS['RMSDMATRIX_DIR'], '{molname}_rmsd.npz')
PATHS['CLUSTERING_OPTIONS_DIR'] = 'clustering_options'
PATHS['CLUSTERING_OPTION_PNG'] = os.path.join(PATHS['CLUSTERING_OPTIONS_DIR'], '{molname}_{eps}.png')
PATHS['BEST_CLUSTERINGS_DIR'] = 'best_clusterings'
PATHS['MEDOID_INDICES_JSON'] = os.path.join(PATHS['BEST_CLUSTERINGS_DIR'], 'medoid_idxs.json')
PATHS['BEST_CLUSTERING_LABELS_JSON'] = os.path.join(PATHS['BEST_CLUSTERINGS_DIR'], '{molname}_labels.json')
PATHS['EMBEDDING_DIR'] = 'embeddings'
PATHS['EMBEDDED_COORDS_DF'] = os.path.join(PATHS['EMBEDDING_DIR'], "{molname}_2dcoords.csv")
PATHS['TOTAL_EMBEDDING_CSV'] = os.path.join(PATHS['EMBEDDING_DIR'], 'embedding_total.csv')
PATHS['REFERENCE_CONFS_DIR'] = 'reference_confs'
PATHS['REFERENCE_CONFS_JSON'] = os.path.join(PATHS['REFERENCE_CONFS_DIR'], '{molname}_refconfs.json')

PATHS['CLUSTERING_EVALUATION_DIR'] = 'clustering_evaluation'
PATHS['CLUSTERING_EVALUATION_PNG'] = os.path.join(PATHS['CLUSTERING_EVALUATION_DIR'], '{molname}_clustering.png')
PATHS['FINAL_DIVERSITY_DIR'] = 'diversity_results'
PATHS['FINAL_DIVERSITY_PNG'] = os.path.join(PATHS['FINAL_DIVERSITY_DIR'], '{molname}_diversity.png')
PATHS['FINAL_DIVERSITY_SVG'] = os.path.join(PATHS['FINAL_DIVERSITY_DIR'], '{molname}_diversity.svg')
PATHS['ENERGYPLOT_DIR'] = 'energy_plots'
PATHS['ENERGYPLOT_PNG'] = os.path.join(PATHS['ENERGYPLOT_DIR'], '{molname}_energies.png')
PATHS['SUMMARY_PDF'] = 'final_report.pdf'

# This is just to trick IDE not to highlight these variables as undefined. What have I done?..
TOTALENSEMBLES_DIR, TOTALENSEMBLE_XYZ, RMSDMATRIX_DIR, CLUSTERING_OPTIONS_DIR, BEST_CLUSTERINGS_DIR, EMBEDDING_DIR, CLUSTERING_OPTION_PNG, MEDOID_INDICES_JSON, EMBEDDED_COORDS_DF, BEST_CLUSTERING_LABELS_JSON, TOTAL_EMBEDDING_CSV, CLUSTERING_EVALUATION_DIR, SUMMARY_PDF, CLUSTERING_EVALUATION_PNG, FINAL_DIVERSITY_DIR, FINAL_DIVERSITY_PNG, FINAL_DIVERSITY_SVG, ENERGYPLOT_DIR, ENERGYPLOT_PNG, REFERENCE_CONFS_DIR, REFERENCE_CONFS_JSON, RMSDMATRIX_FILE = (None,) * len(PATHS._paths)

# Number of threads to use for parallel segments of the workflow
NPROCS = 54

PYXYZ_KWARGS = {
    'mirror_match': True,
    'print_status': False,
}

RINGO_METHOD = 'ringointel'
METHOD_NAMES = {
    RINGO_METHOD: 'Ringo',
    'rdkit': 'ETKDG\n(RDKit)',
    'mtd': 'MTD\n(XTB)',
    'mmbasic': 'MacroModel\nbasic',
    'mmring': 'Macrocyclic\nMacroModel',
    'crest': 'CREST',
}
BIG_TESTS = ['csd_FINWEE10', 'csd_MIWTER', 'csd_RECRAT', 'csd_RULSUN', 'pdb_1NWX', 'pdb_2C6H', 'pdb_2IYA', 'pdb_2QZK', 'pdb_3M6G', 'csd_YIVNOG']
METHODS = {
    'big': [RINGO_METHOD, 'rdkit', 'mtd', 'mmbasic', 'mmring', 'crest'],
    'other': [RINGO_METHOD, 'rdkit', 'mtd'],
}

if 'np' in globals():
    CLUSTERING_EPS_VALUES = tuple(c for c in np.arange(0.1, 0.7, 0.01))

TARGET_REFERENCE_RATIO = {
    # max RMSD => Ratio 1 - n_reference/n_total
    0.4: 0.9,
    0.6: 0.85,
    0.8: 0.7,
    1.0: 0.75,
    1.2: 0.6,
    1.5: 0.5,
    2.0: 0.0,
}
RMSD_PERSPECTIVE_CUTOFF = 0.5

def parse_description(descr):
    descr_parts = descr.split('; ')
    parsed_descr = {
        'method': descr_parts[0],
        'conf_name': descr_parts[1],
        'status': descr_parts[2].split(','),
    }
    assert len(parsed_descr['status']) == 1
    parsed_descr['status'] = parsed_descr['status'][0]
    if len(descr_parts) == 4:
        parsed_descr['energy'] = float(descr_parts[3])
    return parsed_descr

def compose_description(d):
    res = f"{d['method']}; {d['conf_name']}; {d['status']}"
    if 'energy' in res:
        res += f"; {d['energy']}"
    return res

def remove_unperspective(p, molname):
    # Get RMSD matrix from another calculation
    rmsd_file = STARTING_RMSD_MATRIX_FILE.format(molname=molname)
    rmsd_matrix = np.load(rmsd_file)['data']

    lowenergy_idxs = [m.idx for m in p if 'succ' in m.descr]
    def not_unperspective(m):
        cur_index = m.idx
        if cur_index in lowenergy_idxs:
            return True
        
        result = False
        for check_idx in lowenergy_idxs:
            if rmsd_matrix[cur_index, check_idx] < RMSD_PERSPECTIVE_CUTOFF:
                result = True
                break
        return result
    p.filter(not_unperspective)

def merge_ensembles(input_data, args):
    from ringo import Confpool
    PATHS.set_mainwd(args['main_wd'])
    PATHS.load_global()

    molname = input_data['molname']
    print(f"Processing {molname}", flush=True)

    # We remove:
    # 1) Failed optimization 'optfail'
    # 2) Failed topology checks 'topofail'
    # 3) Failed RMSD checks 'rmsdfail'
    # Conformers that stay:
    # 1) Unique conformers within 15 kcal window 'succ'
    # 2) Unique conformers outside of 15 kcal window 'lowerfail' (depends on the execution mode)
    assert not (args['keep_unperspective'] and not args['keep_perspective']), 'Very strange request'
    if args['keep_perspective']:
        allowed_status = ('succ', 'lowerfail')
    else:
        allowed_status = ('succ',)

    check_good = lambda m: parse_description(m.descr)['status'] in allowed_status   

    # Join ensembles obtained with differenct methods into a single ensemble file
    p = Confpool()
    for item in input_data['methods']:
        method = item['method']
        xyzfile = item['xyz']

        before_size = len(p)
        p.include_from_file(xyzfile)
        after_size = len(p)
        for i in range(before_size, after_size):
            # The overall description format is
            # {method}; Conformer {idx}; {status}[; {energy}]
            p[i].descr = f'{method}; {p[i].descr}'
        
    # Remove conformers with 'rmsdfail', etc.
    p.filter(check_good)
    if not args['keep_unperspective'] and args['keep_perspective']:
        remove_unperspective(p, molname)
    
    # Final check of the total ensemble
    resulting_methods = set()
    for m in p:
        conf_data = parse_description(m.descr)
        resulting_methods.add(conf_data['method'])
        assert 'energy' in conf_data
        assert conf_data['status'] in allowed_status

    p.save(input_data['total_xyz'])
    print(f"Saved {input_data['total_xyz']}")

def calc_rmsd_matrix(input_data, args):
    from ringo import Confpool
    PATHS.set_mainwd(args['main_wd'])
    PATHS.load_global()

    print(f"Computing RMSD matrix for {input_data['molname']}", flush=True)

    ensemble_xyz = input_data['ensemble_xyz']
    p = Confpool()
    p.include_from_file(ensemble_xyz)
    p.generate_connectivity(0, mult=1.3)

    graph = p.get_connectivity().copy()
    all_nodes = [i for i in graph.nodes]
    bridges = list(nx.bridges(graph))
    graph.remove_edges_from(bridges)
    # Some of these connected components will be out cyclic parts, others are just single atoms
    components_lists = [list(comp) for comp in nx.connected_components(graph)]

    # Compute separate RMSD matrices with respect to each cyclic part
    rmsd_matrices = []
    for conn_component in components_lists:
        if len(conn_component) == 1:
            continue

        p.generate_connectivity(0, mult=1.3,
                                ignore_elements=[node for node in all_nodes 
                                                 if node not in conn_component])
        cur_graph = p.get_connectivity()
        assert cur_graph.number_of_nodes() == len(conn_component)
        p.generate_isomorphisms()
        matr = p.get_rmsd_matrix(**PYXYZ_KWARGS)
        rmsd_matrices.append(matr)

    first_shape = np.shape(rmsd_matrices[0])
    # Ensure all shapes are the same
    for matrix in rmsd_matrices[1:]:
        assert np.shape(matrix) == first_shape, "Matrices do not have the same shape."

    # Apply element-wise maximum across the matrices
    max_matrix = np.maximum.reduce(rmsd_matrices)
    rmsd_file = input_data['rmsdmatrix_file']
    np.savez_compressed(rmsd_file, data=max_matrix)
    print(f'Saved RMSD matrix {rmsd_file}')

def unite_and_rmsdcalc(args):
    PATHS.set_mainwd(args['main_wd'])
    PATHS.load_global()

    df = pd.read_csv(MAIN_DF_NAME)
    
    unite_tasks = []
    molnames = df['testcase'].unique()
    for molname in molnames:
        methods = df.loc[df['testcase'] == molname, 'method'].unique()
        unite_tasks.append({
            'molname': molname,
            'methods': [
                {
                    'method': method,
                    'xyz': os.path.join(START_XYZ_DIR, f'{molname}_{method}.xyz'),
                }
                for method in methods
            ],
            'total_xyz': TOTALENSEMBLE_XYZ.format(molname=molname),
        })
    parallelize_call(unite_tasks, merge_ensembles, nthreads=NPROCS, args=(args,))
    assert len(glob.glob(os.path.join(TOTALENSEMBLES_DIR, '*.xyz'))) == len(molnames), f"Cannot find {len(molnames)} xyz-files in {os.path.join(TOTALENSEMBLES_DIR, '*.xyz')}"

    rmsdcalc_tasks = [
        {
            'molname': item['molname'],
            'ensemble_xyz': item['total_xyz'],
            'rmsdmatrix_file': RMSDMATRIX_FILE.format(molname=item['molname']),
        }
        for item in unite_tasks
    ]
    parallelize_call(rmsdcalc_tasks, calc_rmsd_matrix, nthreads=NPROCS, args=(args,))


def run_clustering(input_data, args):
    import rpy2.robjects as robjects
    import rpy2.rinterface as rinterface
    rinterface.initr()

    rmsd_file = input_data['rmsd_file']
    molname = input_data['molname']
    print(f'Processing {molname}', flush=True)

    # Uncompress the compressed RMSD matrix
    rmsd_matrix = np.load(rmsd_file)['data']
    rmsd_txt = os.path.join(RMSDMATRIX_DIR, f'{molname}_rmsd.txt')
    np.savetxt(rmsd_txt, rmsd_matrix)

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

    max_eps=2.0
    minpts=5
    cluster_labels = {}
    robjects.r(f"""\
sink("r_output.txt")
library(dbscan)
matrix_data <- as.dist(as.matrix(read.table("{rmsd_txt}")))
res_full <- optics(matrix_data, eps={max_eps}, minPts = {minpts})
""")
    good = tryexec_r(f"""\
res <- extractXi(res_full, xi=0.05)
png("{CLUSTERING_OPTION_PNG.format(molname=molname, eps='auto')}")
plot(res)
dev.off()
""")
    if good:
        try:
            cluster_labels['auto'] = list(robjects.r('res$cluster'))
        except:
            pass

    for cur_eps in CLUSTERING_EPS_VALUES:
        eps_round = round(cur_eps, 2)
        good = tryexec_r(f"""\
res <- extractDBSCAN(res_full, eps_cl = {cur_eps})
png("{CLUSTERING_OPTION_PNG.format(molname=molname, eps=eps_round)}")
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

    with open(input_data['jsonname'], 'w') as f:
        json.dump(cluster_labels, f)

    os.remove(rmsd_txt)

def gen_clustering_plots(args):
    assert not args['use_refconformers'] # This mode uses reference conformers instead of clustering
    
    PATHS.set_mainwd(args['main_wd'])
    PATHS.load_global()
    
    df = pd.read_csv(MAIN_DF_NAME)

    molnames = df['testcase'].unique()
    cluster_tasks = [
        {
            'rmsd_file': RMSDMATRIX_FILE.format(molname=molname),
            'molname': molname,
            'jsonname': os.path.join(CLUSTERING_OPTIONS_DIR, f'{molname}_labels.json')
        }
        for molname in molnames
        if not os.path.isfile(os.path.join(CLUSTERING_OPTIONS_DIR, f'{molname}_labels.json'))
    ]
    # # Do it if it crashes on some testcase
    # bad_cases = ['csd_RULSUN']
    # for item in cluster_tasks:
    #     if not os.path.isfile(item['jsonname']): # item['molname'] in bad_cases and 
    #         run_clustering(item)
    # for item in cluster_tasks:
    #     run_clustering(item, args)
    nthreads = NPROCS
    if nthreads > 12:
        nthreads = 12 # All RAM in the universe won't be enough, if nthreads is bigger
    parallelize_call(cluster_tasks, run_clustering, nthreads=nthreads, args=(args,))
    for item in cluster_tasks:
        assert os.path.isfile(item['jsonname']), f"{item['jsonname']} not found"


def process_good_clusterings(args):
    assert not args['use_refconformers']
    PATHS.set_mainwd(args['main_wd'])
    PATHS.load_global()

    if os.path.isdir(BEST_CLUSTERINGS_DIR):
        shutil.rmtree(BEST_CLUSTERINGS_DIR)
    os.mkdir(BEST_CLUSTERINGS_DIR)

    df = pd.read_csv(MAIN_DF_NAME)
    molnames = df['testcase'].unique()

    best_clusterings = []
    for molname in molnames:
        print(f'Processing {molname}', flush=True)
        json_name = os.path.join(CLUSTERING_OPTIONS_DIR, f'{molname}_labels.json')
        with open(json_name, "r") as f:
            cluster_data = json.load(f)
        
        good_clusterings = []
        for cluster_type, clustering in cluster_data.items():
            clustering = [int(i) for i in clustering]
            if cluster_type == 'auto':
                # clustering = [i if i != 1 else 0 for i in clustering]
                continue
            cluster_types = set(clustering)
            cluster_types.discard(0) # Remove the "cluster" for all unclustered conformations

            num_confs = len(clustering)
            ratio_unclustered = clustering.count(0) / num_confs
            if ratio_unclustered > 0.8 or len(cluster_types) <= 2:
                continue
            
            cluster_sizes = sorted([clustering.count(item)/num_confs for item in cluster_types], reverse=True)
            if cluster_sizes[0] - cluster_sizes[1] > 0.4 or cluster_sizes[1] - cluster_sizes[2] > 0.4:
                continue

            good_clusterings.append({
                'type': cluster_type,
                'score': -ratio_unclustered + len(cluster_types)/50,
            })

        if len(good_clusterings) == 0:
            continue

        best_clustering = max(good_clusterings, key=lambda item: item['score'])
        print(f"{molname} - {best_clustering['type']}")

        best_png = CLUSTERING_OPTION_PNG.format(molname=molname, eps=best_clustering['type'])
        final_png = os.path.join(BEST_CLUSTERINGS_DIR, f"{molname}_{best_clustering['type']}_best.png")
        shutil.copy2(best_png, final_png)
        best_clusterings.append({
            'molname': molname,
            'type': best_clustering['type'],
        })
    
    medoid_idxs_total = {}
    for data in best_clusterings:
        molname = data['molname']
        clustering_type = data['type']
        print(f'Generating cluster medoids for {molname}', flush=True)

        # Load the precomputed RMSD matrix
        rmsd_matrix_fname = RMSDMATRIX_FILE.format(molname=molname)
        rmsd_matrix = np.load(rmsd_matrix_fname)['data']

        # Load labels of the best clustering
        json_name = os.path.join(CLUSTERING_OPTIONS_DIR, f'{molname}_labels.json')
        with open(json_name, 'r') as f:
            cluster_labels = json.load(f)[clustering_type]
        
        # Save best labels for quicker access later
        with open(BEST_CLUSTERING_LABELS_JSON.format(molname=molname), 'w') as f:
            json.dump(cluster_labels, f)

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
        medoid_idxs_total[molname] = medoid_idxs.tolist()
        
        # Create medoid RMSD matrix = submatrix of 'rmsd_matrix'
        medoid_rmsd_matrix = rmsd_matrix[medoid_idxs][:, medoid_idxs]
        medoid_rmsd_fname = os.path.join(BEST_CLUSTERINGS_DIR, f'{molname}_medoidrmsd.npz')
        np.savez_compressed(medoid_rmsd_fname, data=medoid_rmsd_matrix)
    
    with open(MEDOID_INDICES_JSON, 'w') as f:
        json.dump(medoid_idxs_total, f)


def generate_reference_atoms(input_data, args):
    rmsd_file = input_data['rmsd_file']
    molname = input_data['molname']
    result_json = input_data['jsonname']

    print(f'Computing reference atoms for {molname}', flush=True)

    # Load the precomputed RMSD matrix
    rmsd_matrix = np.load(rmsd_file)['data']
    assert rmsd_matrix.shape[0] == rmsd_matrix.shape[1]

    ensemble_size = rmsd_matrix.shape[0]
    random_indices = [i for i in range(ensemble_size)]
    random.shuffle(random_indices)
    
    key_rmsd_values = list(TARGET_REFERENCE_RATIO.keys())
    max_rmsd = max(key_rmsd_values)
    max_target = TARGET_REFERENCE_RATIO[max_rmsd]
    def get_target_ratio(rmsd):
        if rmsd >= max_rmsd:
            return max_target
        key_rmsd = min((
            v
            for v in key_rmsd_values
            if v > rmsd
        ))
        return TARGET_REFERENCE_RATIO[key_rmsd]
    get_ratio = lambda n_ref, n_confs: 1 - n_ref/n_confs
    
    reference_atoms = [i for i in range(ensemble_size)] # To enter the while-loop
    chosen_rmsd_cutoff = 0.1
    while get_ratio(len(reference_atoms), ensemble_size) < get_target_ratio(chosen_rmsd_cutoff):
        chosen_rmsd_cutoff += 0.02

        reference_atoms = []
        for conf_index in random_indices:
            covered = False
            for ref_index in reference_atoms:
                if rmsd_matrix[conf_index, ref_index] < chosen_rmsd_cutoff:
                    covered = True
                    break
            if not covered:
                reference_atoms.append(conf_index)
        # print(f'RMSD={chosen_rmsd_cutoff} Ratio={get_ratio(len(reference_atoms), ensemble_size)} ... {get_target_ratio(chosen_rmsd_cutoff)}')
    
    # print(f'Chosen RMSD={chosen_rmsd_cutoff} with ratio={round(get_ratio(len(reference_atoms), ensemble_size), 2)}')

    group_to_confs = {
        int(ref_index): [
            conf_index
            for conf_index in range(ensemble_size)
            if rmsd_matrix[conf_index, ref_index] < chosen_rmsd_cutoff
        ]
        for ref_index in reference_atoms
    }

    conf_to_groups = {
        int(conf_index): [
            ref_index
            for ref_index in reference_atoms
            if rmsd_matrix[conf_index, ref_index] < chosen_rmsd_cutoff
        ]
        for conf_index in range(ensemble_size)
    }

    with open(result_json, 'w') as f:
        json.dump({
            'conf_to_groups': conf_to_groups,
            'group_to_confs': group_to_confs,
            'rmsd_cutoff': chosen_rmsd_cutoff
        }, f)


def do_confspace_partition(args):
    assert args['use_refconformers']
    PATHS.set_mainwd(args['main_wd'])
    PATHS.load_global()

    df = pd.read_csv(MAIN_DF_NAME)
    molnames = df['testcase'].unique()

    refatoms_tasks = [
        {
            'rmsd_file': RMSDMATRIX_FILE.format(molname=molname),
            'molname': molname,
            'jsonname': REFERENCE_CONFS_JSON.format(molname=molname),
        }
        for molname in molnames
    ]
    parallelize_call(refatoms_tasks, generate_reference_atoms, nthreads=NPROCS, args=(args,))
    assert len(glob.glob(REFERENCE_CONFS_JSON.format(molname='*'))) == len(molnames)

    # These are not really medoids of any clusters but we'll store them like this
    medoid_idxs_total = {}
    for item in refatoms_tasks:
        with open(item['jsonname'], 'r') as f:
            group_to_confs = json.load(f)['group_to_confs']
        medoid_idxs_total[item['molname']] = [int(key) for key in group_to_confs.keys()]
            
    with open(MEDOID_INDICES_JSON, 'w') as f:
        json.dump(medoid_idxs_total, f)


def run_embedding(input_data, args):
    from sklearn.manifold import TSNE

    rmsd_file = input_data['rmsd_file']
    molname = input_data['molname']
    embedding_file = input_data['embedding_file']
    labels_file = input_data['labels_file']

    print(f'Doing embedding for {molname}', flush=True)

    if not os.path.isfile(embedding_file):
        # Load the precomputed RMSD matrix
        rmsd_matrix = np.load(rmsd_file)['data']
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

    else:
        df = pd.read_csv(embedding_file)
        embedding_xy = [(x, y) for x, y in zip(df['x'], df['y'])]
    
    # Load clustering labels
    with open(labels_file, 'r') as f:
        cluster_labels = json.load(f)
    if args['use_refconformers']:
        cluster_labels = cluster_labels['conf_to_groups']
        cluster_labels = {int(key): value for key, value in cluster_labels.items()}
        cluster_labels = [tuple(cluster_labels[i]) for i in range(len(cluster_labels))]

    # Save the dataframe of embedded conformers
    df_embedded = pd.DataFrame(embedding_xy, columns=['x', 'y'])
    df_embedded['cluster'] = cluster_labels
    
    # Create a boolean column indicating whether the structure is a medoid of a cluster
    with open(MEDOID_INDICES_JSON, 'r') as f:
        medoid_idxs_total = json.load(f) # Load medoid indices
    df_embedded['is_medoid'] = df_embedded.index.isin(medoid_idxs_total[molname])
    
    # Save CSV. Keep indices in separate column
    df_embedded.to_csv(embedding_file, index_label='conf_id')

def calc_2d_embeddings(args):
    PATHS.set_mainwd(args['main_wd'])
    PATHS.load_global()

    df = pd.read_csv(MAIN_DF_NAME)
    molnames = df['testcase'].unique()

    def get_labels_filename(molname):
        if args['use_refconformers']:
            return REFERENCE_CONFS_JSON.format(molname=molname)
        else:
            return BEST_CLUSTERING_LABELS_JSON.format(molname=molname)

    embedding_tasks = [
        {
            'molname': molname,
            'rmsd_file': RMSDMATRIX_FILE.format(molname=molname),
            'embedding_file': EMBEDDED_COORDS_DF.format(molname=molname),
            'labels_file': get_labels_filename(molname),
        }
        for molname in molnames
        # We don't do clustering for all test molecules (only when the result is reasonable)
        if os.path.isfile(get_labels_filename(molname))
    ]
    # scikit can't control the number of threads it uses, so...
    for task in embedding_tasks:
        run_embedding(task, args)

    # Merge 'embedding_file' dataframes into a single one
    dfs = []
    for data in embedding_tasks:
        embedding_file = data['embedding_file']
        assert os.path.isfile(embedding_file)
        subdf = pd.read_csv(embedding_file)
        subdf['testcase'] = data['molname']
        dfs.append(subdf)
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(TOTAL_EMBEDDING_CSV, index=False)

def get_points_df(molname):
    import pyxyz

    p = pyxyz.Confpool()
    p.include_from_file(TOTALENSEMBLE_XYZ.format(molname=molname))
    
    df = pd.read_csv(TOTAL_EMBEDDING_CSV)
    df = df[df['testcase'] == molname]
    
    # Record methods that produced each conformation
    methods_list = [m.descr.split('; ')[0] for m in p]
    df['method'] = methods_list
    df['method'] = df['method'].astype('category')

    # Get energy data
    df['status'] = [m.descr.split('; ')[2] for m in p]
    df['energy'] = [float(m.descr.split('; ')[3]) for m in p]
    df['is_lowenergy'] = df['status'] == 'succ'

    lowenergy_idxs = df[df['is_lowenergy']]['conf_id'].tolist()
    rmsd_matrix = np.load(RMSDMATRIX_FILE.format(molname=molname))['data']
    def check_perspective(item):
        if item['is_lowenergy']:
            return False
        index = item['conf_id']

        result = False
        for check_idx in lowenergy_idxs:
            if rmsd_matrix[index, check_idx] < RMSD_PERSPECTIVE_CUTOFF:
                result = True
                break
        return result
    df['is_perspective'] = df.apply(check_perspective, axis=1)
    return df

def get_cluster_df(point_df):
    # Create df of cluster medoids
    medoid_df = point_df[point_df['is_medoid']] \
                .drop(['is_medoid', 'method'], axis=1) \
                .copy()
    
    # Create df of methods that found each cluster
    clusters = point_df['cluster'].unique().tolist()
    all_methods = point_df['method'].unique().tolist()
    cluster_found_data = {cluster: {} for cluster in clusters}
    for cur_cluster in clusters:
        methods_found = point_df[point_df['cluster'] == cur_cluster]['method'].unique().tolist()
        cluster_found_data[cur_cluster] = {
            f'{method}_found': method in methods_found
            for method in all_methods
        }
    found_df = pd.DataFrame.from_dict(cluster_found_data, orient='index')
    found_df.index.name = 'cluster'
    found_df = found_df.reset_index()

    # Assemble found-data and mean points in a single df
    cluster_df = pd.merge(medoid_df, found_df, on='cluster')
    # Get rid of cluster of unassigned points
    cluster_df = cluster_df[cluster_df['cluster'] > 0]
    return cluster_df

def get_refpoint_df(point_df, molname, args):
    assert not args['expand_clusters'], "'expand_clusters' is not implemented for refpoints"

    # Create df of cluster medoids
    medoid_df = point_df[point_df['is_medoid']] \
                .drop(['is_medoid', 'method'], axis=1) \
                .copy()
    
    # Load the coverage data (how points relate to vicinities of refpoints)
    with open(REFERENCE_CONFS_JSON.format(molname=molname), 'r') as f:
        cluster_assignments = json.load(f)
    cluster_assignments = cluster_assignments['group_to_confs']
    cluster_assignments = {int(key): value for key, value in cluster_assignments.items()}

    # Create df of methods that found each cluster
    clusters = list(point_df[point_df['is_medoid']]['conf_id']) # In 'perspective' mode, the cluster idx are the indices of refponts
    all_methods = point_df['method'].unique().tolist()
    cluster_found_data = {cluster: {} for cluster in clusters}
    
    if args['plot_highenergy'] and args['plot_perspective']:
        allowed_df = point_df
    elif args['plot_highenergy'] and not args['plot_perspective']:
        raise ValueError('Very questionable request: plot_highenergy=True and plot_perspective=False')
    elif not args['plot_highenergy'] and args['plot_perspective']:
        allowed_df = point_df[point_df['is_lowenergy'] | point_df['is_perspective']]
    elif not args['plot_highenergy'] and not args['plot_perspective']:
        allowed_df = point_df[point_df['is_lowenergy']]

    for cur_cluster in clusters:
        covered_conformers = cluster_assignments[cur_cluster]
        methods_found = allowed_df[allowed_df['conf_id'].isin(covered_conformers)]['method'].unique().tolist()
        cluster_found_data[cur_cluster] = {
            f'{method}_found': method in methods_found
            for method in all_methods
        }
    found_df = pd.DataFrame.from_dict(cluster_found_data, orient='index')
    found_df.index.name = 'cluster'
    found_df = found_df.reset_index()

    # Fix the 'cluster' column of 'medoid_df' for merging with 'found_df'
    import ast
    def fix_cluster_column(cluster_group):
        cluster_tuple = ast.literal_eval(cluster_group)
        assert len(cluster_tuple) == 1
        return cluster_tuple[0]
    medoid_df['cluster'] = medoid_df['cluster'].apply(fix_cluster_column)

    # Assemble found-data and mean points in a single df
    cluster_df = pd.merge(medoid_df, found_df, on='cluster')
    return cluster_df

def get_combined_df(molname, args):
    # Get df of 2D coordinates for each conformer + method, cluster
    point_df = get_points_df(molname)
    
    # Get df of mean 2D coords for each cluster + methods that found each cluster
    if args['use_refconformers']:
        cluster_df = get_refpoint_df(point_df, molname, args)
    else:
        cluster_df = get_cluster_df(point_df)

    point_df = point_df.drop(['method', 'is_medoid'], axis=1).reset_index(drop=True)
    
    if args['use_refconformers']:
        import ast
        def leave_single_cluster(cluster_group):
            cluster_tuple = ast.literal_eval(cluster_group)
            return cluster_tuple[0]
        point_df['cluster'] = point_df['cluster'].apply(leave_single_cluster)

    facet_parts = []
    plottype_order = []

    # Prepare raw points for combined drawing
    rawpoints_df = point_df.copy()
    rawpoints_df['type'] = 'raw_points'
    facet_parts.append(rawpoints_df)
    plottype_order.append('raw_points')
    
    # Prepare cluster centers for combined drawing. Remove '{method}_found' columns
    cluster_medoids_df = cluster_df.filter(items=['cluster', 'x', 'y', 'energy'], axis=1).reset_index(drop=True)
    cluster_medoids_df['type'] = 'cluster_centers'
    cluster_medoids_df = cluster_medoids_df[cluster_medoids_df['cluster'] > 0]
    facet_parts.append(cluster_medoids_df)
    plottype_order.append('cluster_centers')

    if not args['use_refconformers']:
        cluster_points = point_df.copy()
        cluster_points['type'] = 'cluster_points'
        cluster_points = cluster_points[cluster_points['cluster'] > 0]
        facet_parts.append(cluster_points)
        plottype_order.append('cluster_points')
    
    if args['plot_highenergy']:
        high_energy_points = point_df.copy()
        high_energy_points['type'] = 'high_energy_points'
        high_energy_points = high_energy_points[(~high_energy_points['is_lowenergy']) & ((~high_energy_points['is_perspective']))]
        facet_parts.append(high_energy_points)
        plottype_order.append('high_energy_points')
    
    if args['plot_perspective']:
        perspective_points = point_df.copy()
        perspective_points['type'] = 'perspective_points'
        perspective_points = perspective_points[perspective_points['is_perspective']]
        facet_parts.append(perspective_points)
        plottype_order.append('perspective_points')

    low_energy_points = point_df.copy()
    low_energy_points['type'] = 'low_energy_points'
    low_energy_points = low_energy_points[low_energy_points['is_lowenergy']]
    facet_parts.append(low_energy_points)
    plottype_order.append('low_energy_points')
    
    # Build df for combined plot
    from pandas.api.types import CategoricalDtype
    combined_df = pd.concat(facet_parts) \
                    .reset_index(drop=True)
    combined_df['cluster'] = combined_df['cluster'].astype('category')
    facet_type = CategoricalDtype(categories=plottype_order, ordered=True)   
    combined_df['type'] = combined_df['type'].astype(facet_type)
    return combined_df

def get_coverage_ratio(testcase):
    with open(REFERENCE_CONFS_JSON.format(molname=testcase), 'r') as f:
        cluster_assignments = json.load(f)
    num_groups = len(cluster_assignments['group_to_confs'])
    num_confs = len(cluster_assignments['conf_to_groups'])
    rmsd_cutoff = cluster_assignments['rmsd_cutoff']
    return 1 - num_groups/num_confs, rmsd_cutoff

def plot_clustering_thread(input_data, args):
    from plotnine import ggplot, aes, geom_point, labs, facet_grid, theme_bw, element_text, theme, element_blank, element_rect, element_line, scale_size_manual
    iteration = input_data['iteration']
    testcase = input_data['testcase']
    num_iterations = input_data['num_iterations']

    print(f"Doing clustering evaluation for {testcase} ({iteration}/{num_iterations})", flush=True)
    # Get df for combined plot
    combined_df = get_combined_df(testcase, args)

    lowest_energy = combined_df['energy'].min()
    combined_df['energy'] = combined_df['energy'] - lowest_energy

    if args['use_refconformers']:
        cover_ratio, rmsd_cutoff = get_coverage_ratio(testcase)
        additional_data = f' (points coverage = {round(cover_ratio * 100, 2)}%, RMSD cutoff = {rmsd_cutoff:.2f})'
    else:
        additional_data = ''

    # Prerequisites for plotnine plots
    facet_theme = (theme_bw() +
        theme(panel_grid_major = element_blank(),
        panel_grid_minor = element_blank(),
        panel_border = element_rect(colour="black", fill=None, size=1),
        axis_line = element_line(colour="black"),
        axis_title = element_text(size=12, face="bold"),
        axis_text = element_text(size=14),
        legend_title = element_text(size=14, face="bold"),
        legend_text = element_text(size=14),
        strip_text=element_text(size=14, face="bold")))

    remove_axes = theme(axis_text_x=element_blank(),
        axis_text_y=element_blank(),
        axis_ticks=element_blank(),
        axis_line=element_blank(),
        axis_title_y=element_blank())

    rename_facet = {
        'raw_points': 'Full\nensemble',
        'cluster_points': 'Only clustered\nconformers',
        'cluster_centers': 'Cluster\ncenters',
        'low_energy_points': 'Low\nenergy',
        'high_energy_points': 'High\nenergy',
        'perspective_points': 'Perspective\nconformers',
    }
    combined_df.replace(rename_facet, inplace=True)

    point_size_custom = {
        'raw_points': 0.6,
        'cluster_points': 0.6,
        'low_energy_points': 0.6,
        'high_energy_points': 0.6,
        'perspective_points': 0.6,
        'cluster_centers': 0.6 # 2
    }
    point_size_custom = {rename_facet[key]: value for key, value in point_size_custom.items()}
    
    plot = ggplot(combined_df, aes(x='x', y='y', color='energy', size='type')) + geom_point() \
        + facet_theme + remove_axes + theme(figure_size=(10, 4)) + facet_grid('~type') \
        + scale_size_manual(values=point_size_custom) \
        + labs(title=f'Cumulative ensemble summary{additional_data}', x = "")
    plot.save(CLUSTERING_EVALUATION_PNG.format(molname=testcase), verbose=False)

def plot_clustering_quality(args):
    PATHS.set_mainwd(args['main_wd'])
    PATHS.load_global()

    embedding_df = pd.read_csv(TOTAL_EMBEDDING_CSV)
    testcases = embedding_df['testcase'].unique()

    plotting_tasks = [
        {
            'testcase': testcase,
            'iteration': i,
            'num_iterations': len(testcases),
        }
        for i, testcase in enumerate(testcases)
    ]
    parallelize_call(plotting_tasks, plot_clustering_thread, nthreads=NPROCS, args=(args,))
    assert len(glob.glob(CLUSTERING_EVALUATION_PNG.format(molname='*'))) == len(plotting_tasks)


def get_medoid_df(molname, args):
    point_df = get_points_df(molname)
    medoid_coord_df = pd.read_csv(EMBEDDED_COORDS_DF.format(molname=molname))
    medoid_coord_df = medoid_coord_df[medoid_coord_df['is_medoid']].reset_index()

    # Create df of methods that found each cluster
    clusters = point_df['cluster'].unique().tolist()
    all_methods = point_df['method'].unique().tolist()
    cluster_found_data = {cluster: {} for cluster in clusters}
    for cur_cluster in clusters:
        methods_found = point_df[point_df['cluster'] == cur_cluster]['method'].unique().tolist()
        cluster_found_data[cur_cluster] = {
            f'{method}_found': method in methods_found
            for method in all_methods
        }
    found_df = pd.DataFrame.from_dict(cluster_found_data, orient='index')
    found_df.index.name = 'cluster'
    found_df = found_df.reset_index()

    # Assemble found-data and mean points in a single df
    if args['expand_clusters']:
        res_df = pd.merge(point_df, found_df, on='cluster')
        
        # Highlight unassigned points
        res_df['unclustered'] = res_df['cluster'] == 0
    else:
        res_df = pd.merge(medoid_coord_df, found_df, on='cluster')

        # Get rid of cluster of unassigned points
        res_df = res_df[res_df['cluster'] > 0].reset_index(drop=True)
    
    return res_df

def get_relation(ringo_found, other_found, unclustered=None, methods=None, ref_method=None):
    # Check if both arrays have the same length
    if len(ringo_found) != len(other_found):
        raise ValueError("Arrays must have the same length.")
    
    # Create a new array to store the combined values
    combined_arr = np.empty(len(ringo_found), dtype=object)
    
    # Iterate over the arrays and assign 'both' to the corresponding element if both conditions are True
    for i in range(len(ringo_found)):
        if unclustered is not None and unclustered[i]:
            if methods[i] == RINGO_METHOD:
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

def do_ggplot(plotdf, key_names, res_fname, custom_order):
    import rpy2.robjects as robjects
    import rpy2.rinterface as rinterface
    from rpy2.robjects import pandas2ri
    from scipy.spatial.distance import cdist

    # Restart the R session
    rinterface.initr()
    pandas2ri.activate()
    
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
        key_names['ringo']: '#66a61e', # Green
        key_names['other']: '#d95f02', # Orange
        key_names['both']: '#7570b3', # Blue
        key_names['ringo_unclustered']: '#777777', # Grey
        key_names['other_unclustered']: '#000000', # Black
        key_names['other_explored']: '#e6ab02', # Yellow
        # 'total_explored': '#ffffff', # white
    }
    r_custom_colors = robjects.vectors.ListVector(custom_colors)
    robjects.globalenv['custom_colors'] = r_custom_colors
    
    # Pass point sizes into R
    custom_sizes = {
        key_names['ringo']: 0.7,
        key_names['other']: 0.7,
        key_names['both']: 0.7,
        key_names['ringo_unclustered']: 0.3,
        key_names['other_unclustered']: 0.3,
    }
    r_custom_sizes = robjects.vectors.ListVector(custom_sizes)
    robjects.globalenv['custom_sizes'] = r_custom_sizes
    
    # Pass the desired order of point layering
    r_custom_order = robjects.vectors.StrVector(list(reversed(custom_order)))
    robjects.globalenv['custom_order'] = r_custom_order

    # Pass the density df
    densitydf_total = plotdf.copy()
    densitydf_total['pointtype'] = 'total_explored'
    densitydf_alt = plotdf.copy()
    densitydf_alt = densitydf_alt[(densitydf_alt['Who found?'] != key_names['ringo']) &\
                                  (densitydf_alt['Who found?'] != key_names['ringo_unclustered'])]
    densitydf_alt['pointtype'] = key_names['other_explored']
    densitydf  = densitydf_alt # pd.concat([], ignore_index=True)
    densitydf = densitydf[['x', 'y', 'alt_method']].reset_index(drop=True)
    
    methods = plotdf['alt_method'].unique()
    leveldfs = []
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

    contour_isovalues = [0.03,]
    # Pass the desired order of point layering
    r_contour_isovalues = robjects.vectors.FloatVector(list(reversed(contour_isovalues)))
    robjects.globalenv['contour_isovalues'] = r_contour_isovalues
    
    r_leveldf = pandas2ri.py2rpy(leveldf)
    robjects.globalenv['leveldf'] = r_leveldf
    
    def plot(safe=False):
        # Execute R code
        if safe:
            optional_density = ""
        else:
            optional_density = f"geom_contour(leveldf, mapping = aes(x = x, y = y, z = z), colour='#e6ab02', breaks = contour_isovalues) +"
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
    plotdf$`Who found?` <- factor(plotdf$`Who found?`, levels = custom_order)
    plotdf <- plotdf[order(plotdf$`Who found?`), ]
    
    plot <- ggplot() + {optional_density}
        geom_point(plotdf, mapping = aes(x = x, y = y, color = `Who found?`, size = `Who found?`)) +
        scale_size_manual(values = custom_sizes_values) +
        facet_grid(. ~ alt_method) +
        facet_theme + remove_axes +
        scale_color_manual(values = custom_colors) +
        labs(x = '', title='Diversity benchmark') +
        theme(text=element_text(family="Arial"))
    ggsave(plot, filename = "{res_fname}", width = {plot_width}, height = 3)
    """
        robjects.r(rcode)

    try:
        plot()
    except:
        plot(safe=True)

def get_method_timings(testcase):
    df = pd.read_csv(MAIN_DF_NAME)
    df = df[df['testcase'] == testcase]
    timings = {
        method: time * thread_avg
        for method, time, thread_avg in zip(df['method'], df['time'], df['thread_avg'])
    }
    return timings

def plot_diversity_thread(input_data, args):
    iteration = input_data['iteration']
    testcase = input_data['testcase']
    num_iterations = input_data['num_iterations']
    
    print(f"Evaluating diversity for {testcase} ({iteration}/{num_iterations})", flush=True)

    if args['use_refconformers']:
        df = get_refpoint_df(get_points_df(testcase), testcase, args)
    else:
        df = get_medoid_df(testcase,args) # TODO CHECK CAREFULLY!!!

    other_methods = tuple(method for method in METHOD_NAMES.keys() if method != RINGO_METHOD and f'{method}_found' in df)
    dfs = {}
    for method in other_methods:
        if 'unclustered' in df:
            assert 'method' in df
            kwargs = {'methods': df['method'], 'unclustered': df['unclustered'], 'ref_method': method}
        else:
            kwargs = {}
        df[f'relation_{method}'] = get_relation(df[f'{RINGO_METHOD}_found'], df[f'{method}_found'], **kwargs)

    # Prepare dfs for particular figures
    key_names = {
        'ringo': 'Ringo (only)',
        'other': 'Alternative (only)',
        'both': 'Both',
        'ringo_unclustered': 'Ringo (unclustered)',
        'other_unclustered': 'Alternative (unclustered)',
        'other_explored': 'Space explored by alternative',
    }
    
    timings = get_method_timings(testcase)

    for method in other_methods:
        remaining_methods = [m for m in other_methods if m != method]
        drop_axes = []
        for m in remaining_methods:
            drop_axes.append(f'{m}_found')
            drop_axes.append(f'relation_{m}')
        drop_axes.append(f'{RINGO_METHOD}_found')
        drop_axes.append('cluster')
        drop_axes.append(f'{method}_found')

        # Define the custom order of categories
        custom_order = [
            key_names['ringo'],
            key_names['both'],
            key_names['other'],
            key_names['ringo_unclustered'],
            key_names['other_unclustered'],
        ]

        dfs[method] = df.drop(drop_axes, axis=1) \
                        .dropna() \
                        .rename(columns={f'relation_{method}': 'Who found?'}) \
                        .replace(key_names)
        # Convert the column to a categorical type with the custom order
        dfs[method]['Who found?'] = dfs[method]['Who found?'].astype(pd.CategoricalDtype(categories=custom_order, ordered=True))

        times_speed = timings[method] / timings[RINGO_METHOD]
        if str(round(times_speed, 1)) != '1.0': 
            if times_speed > 1.0:
                speed_word = 'slower'
            else:
                times_speed = 1 / times_speed
                speed_word = 'faster'
            speed_line = f'\n({times_speed:.1f}x {speed_word})'
        else:
            speed_line = ''
        dfs[method]['alt_method'] = METHOD_NAMES[method] + speed_line
        dfs[method] = dfs[method].sort_values(by='Who found?')


    # To plot facet_grid we need a single df
    plotdf = pd.concat([df for df in dfs.values()], ignore_index=True)

    result_png = FINAL_DIVERSITY_PNG.format(molname=testcase)
    do_ggplot(plotdf, key_names, result_png, custom_order)

    # result_svg = FINAL_DIVERSITY_SVG.format(molname=testcase)
    # do_ggplot(plotdf, key_names, result_svg, custom_order)

def plot_diversity(args):
    import pyxyz
    PATHS.set_mainwd(args['main_wd'])
    PATHS.load_global()
    
    # Find all molecules with embedded medoids
    embedding_df = pd.read_csv(TOTAL_EMBEDDING_CSV)
    testcases = embedding_df['testcase'].unique()

    plotting_tasks = []
    for testcase in testcases:
        print(f'Checking out {testcase}')
        p = pyxyz.Confpool()
        p.include_from_file(TOTALENSEMBLE_XYZ.format(molname=testcase))
    
        ringo_found = False
        for m in p:
            if RINGO_METHOD in m.descr:
                ringo_found = True
                break
        if not ringo_found:
            continue

        plotting_tasks.append({
            'testcase': testcase,
            'num_iterations': len(testcases),
        })
    for i in range(len(plotting_tasks)):
        plotting_tasks[i]['iteration'] = i
        
    parallelize_call(plotting_tasks, plot_diversity_thread, nthreads=NPROCS, args=(args,))
    assert len(glob.glob(FINAL_DIVERSITY_PNG.format(molname='*'))) == len(plotting_tasks)


def plot_energies_thread(input_data, args):
    from plotnine import ggplot, aes, ggtitle, geom_point, labs, facet_grid, theme_bw, element_text, theme, element_blank, element_rect, element_line, scale_size_manual, scale_shape_manual
    
    iteration = input_data['iteration']
    testcase = input_data['testcase']
    num_iterations = input_data['num_iterations']

    print(f"Making energry plots for {testcase}: ({iteration}/{num_iterations})", flush=True)

    df = get_points_df(testcase)
    lowest_energy = df['energy'].min()
    df['energy'] = df['energy'] - lowest_energy

    def get_energy_status(item):
        if item['is_lowenergy']:
            assert not item['is_perspective']
            return 'lowenergy'
        elif item['is_perspective']:
            return 'perspective'
        else:
            return 'unperspective'
    df['energy_status'] = df.apply(get_energy_status, axis=1)

    energy_status_map = {
        'lowenergy': 'Low energy (<15 kcal/mol)',
        'perspective': 'Perspective, >15 kcal/mol',
        'unperspective': 'Unperspective, >15 kcal/mol',
    }
    df['energy_status'] = df['energy_status'].replace(energy_status_map)

    colnames = {
        'energy_status': 'Energy status',
        'energy': 'Energy, kcal/mol'
    }
    df = df.rename(columns=colnames)
    df = df.replace(METHOD_NAMES)

    facet_theme = (theme_bw() +
                theme(panel_grid_major = element_blank(),
                panel_grid_minor = element_blank(),
                panel_border = element_rect(colour="black", fill=None, size=1),
                axis_line = element_line(colour="black"),
                axis_title = element_text(size=12, face="bold"),
                axis_text = element_text(size=14),
                legend_title = element_text(size=14, face="bold"),
                legend_text = element_text(size=14),
                strip_text=element_text(size=14, face="bold")))
    
    remove_axes = theme(axis_text_x=element_blank(),
                axis_text_y=element_blank(),
                axis_ticks=element_blank(),
                axis_line=element_blank(),
                axis_title_y=element_blank())
    
    point_size_custom = {
        energy_status_map['lowenergy']: 1.0,
        energy_status_map['perspective']: 0.2,
        energy_status_map['unperspective']: 0.1,
    }
    shapes_custom = {
        energy_status_map['lowenergy']: 'x',
        energy_status_map['perspective']: 'o',
        energy_status_map['unperspective']: 'o',
    }
    number_of_methods = len(df['method'].unique())
    if number_of_methods > 3:
        figure_width = 13
    else:
        figure_width = 10

    plot = ggplot(df, aes(x='x', y='y', color=colnames['energy'], shape=colnames['energy_status'], size=colnames['energy_status'])) \
                + geom_point() \
                + scale_size_manual(values=point_size_custom) \
                + facet_theme + remove_axes + theme(figure_size=(figure_width, 4)) + facet_grid('~method') \
                + scale_shape_manual(values=shapes_custom) \
                + labs(title=f'Energy summary by methods', x = "")
    png_name = ENERGYPLOT_PNG.format(molname=testcase)
    plot.save(png_name, verbose=False, dpi=500)

def plot_energies(args):
    PATHS.set_mainwd(args['main_wd'])
    PATHS.load_global()

    embedding_df = pd.read_csv(TOTAL_EMBEDDING_CSV)
    testcases = embedding_df['testcase'].unique()

    plotting_tasks = [
        {
            'testcase': testcase,
            'iteration': i,
            'num_iterations': len(testcases),
        }
        for i, testcase in enumerate(testcases)
    ]
    parallelize_call(plotting_tasks, plot_energies_thread, nthreads=NPROCS, args=(args,))
    assert len(glob.glob(ENERGYPLOT_PNG.format(molname='*'))) == len(plotting_tasks)


def pil_to_reportlab(pil_image):
    import io
    from reportlab.lib.utils import ImageReader
    
    side_im_data = io.BytesIO()
    pil_image.save(side_im_data, format='png')
    side_im_data.seek(0)
    return ImageReader(side_im_data)

def pillow_text(text, big=False, align='left'):
    from chemscripts.imageutils import TrimmingBox
    from PIL import Image, ImageDraw, ImageFont

    # Create a new image
    width, height = 3000, 1000
    background_color = (255, 255, 255)
    image = Image.new("RGB", (width, height), background_color)

    # Add text to the image
    draw = ImageDraw.Draw(image)
    text_color = (0, 0, 0)
    font_size = 36
    if big:
        font_size = 42
    font = ImageFont.truetype("./arial.ttf", font_size)
    text_position = (50, 50)
    draw.text(text_position, text, fill=text_color, font=font, align=align)

    maximal_box = TrimmingBox()
    maximal_box.extend(image)
    image = image.crop(maximal_box.points)
    return image

def resize_to_width(im, desired_width):
    from PIL import Image
    aspect_ratio = im.width / im.height
    desired_height = int(desired_width / aspect_ratio)
    im.thumbnail((desired_width, desired_height), Image.ANTIALIAS)

def assemble_summary(args):
    from chemscripts.imageutils import HImageLayout, VImageLayout
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from PIL import Image

    PATHS.set_mainwd(args['main_wd'])
    PATHS.load_global()
    
    embedding_df = pd.read_csv(TOTAL_EMBEDDING_CSV)
    testcases = embedding_df['testcase'].unique()

    c = canvas.Canvas(SUMMARY_PDF, pagesize=letter)
    testcases_seq = [testcase for testcase in testcases if testcase in BIG_TESTS]
    messages = {
        0: f'Molecules tested with CREST and MacroModel\n({len(BIG_TESTS)} molecules)',
        len(testcases_seq): f'The rest of the test set\n({len(testcases) - len(BIG_TESTS)} molecules)',
    }
    testcases_seq += [testcase for testcase in testcases if testcase not in BIG_TESTS]

    for idx, testcase in enumerate(testcases_seq):
        print(f'Processing {testcase}', flush=True)
        evalpng_path = CLUSTERING_EVALUATION_PNG.format(molname=testcase)
        if not args['use_refconformers']:
            cutoffpng_path = glob.glob(os.path.join(BEST_CLUSTERINGS_DIR, f'{testcase}_*_best.png'))[0]
        energyplot_path = ENERGYPLOT_PNG.format(molname=testcase)
        benchplot_path = FINAL_DIVERSITY_PNG.format(molname=testcase)
        
        evalpng = Image.open(evalpng_path)
        if not args['use_refconformers']:
            cutoffpng = Image.open(cutoffpng_path)
        energyplot = Image.open(energyplot_path)
        loaded_bench = os.path.isfile(benchplot_path)
        if loaded_bench:
            benchplot = Image.open(benchplot_path)

        main_image = VImageLayout()
        left_side = VImageLayout()
        left_side.insert(pillow_text(f"Results for '{testcase}'"), type='middle')
        left_side.insert(evalpng, type='middle')
        top_part = HImageLayout()
        top_part.insert(left_side.build(), type='middle')
        if not args['use_refconformers']:
            top_part.insert(cutoffpng, type='middle')

        top_part_img = top_part.build()
        resize_to_width(energyplot, top_part_img.width)
        if loaded_bench:
            resize_to_width(benchplot, top_part_img.width)

        if idx in messages:
            main_image.insert(pillow_text(messages[idx], big=True, align='center'), type='middle')
        main_image.insert(top_part_img, type='middle')
        main_image.insert(energyplot, type='middle')
        if loaded_bench:
            main_image.insert(benchplot, type='middle')
        else:
            main_image.insert(pillow_text("Ringo couldn't find any perspective/low-energy conformers", big=True, align='center'), type='middle')


        main_image = main_image.build()
        c.setPageSize(main_image.size)
        c.drawImage(pil_to_reportlab(main_image), 0, 0, width=main_image.width, height=main_image.height)
        c.showPage()
    c.save()
    print(f'Results are written in file {SUMMARY_PDF}')

if __name__ == "__main__":
    import environments as env

    """
    keys in run_settings specify the important aspects of diversity analysis and
    the way of results visualization. This keys have the following meanings:

    * main_wd - working directory for the entire diversity analysis
    * mode - name of the execution mode. It's not used anywhere, but it can be for some hardcoded branches in analysis logic.
    * keep_unperspective - drop all unperspective conformers at the beginning of the analysis
    * keep_perspective - drop all perspective high-energy conformers at the beginning of the analysis
    * use_refconformers - disable clustering and use reference conformers instead
    * plot_highenergy - account for unperspective conformers when deciding which method found a cluster
    * plot_perspective - account for perspective high-energy conformers when deciding which method found a cluster
    * expand_clusters - generate diversity plots 
    """

    # This is the mode of diversity analysis used in the paper
    # filtered conformers (remove unperspective from the start, employ clustering, plot conformers instead of clusters)
    run_settings = {
        'main_wd': './filtered_ensemble_diversity',
        'mode': 'filtered',
        'keep_unperspective': False,
        'keep_perspective': True,
        'use_refconformers': False,
        'plot_highenergy': True, # We removed them already, so this disables the redundant check for perspective conformers during plotting
        'plot_perspective': True,
        'expand_clusters': True,
    }

    # Tasks to be done
    # 1) Compute RMSD matrices on ring atoms
    env.exec(__file__, func=unite_and_rmsdcalc, env='intel', args=[run_settings])
    
    # 2.1) Obtain optimal conformer clusterings
    if run_settings['use_refconformers']:
        env.exec(__file__, func=do_confspace_partition, env='intel', args=[run_settings])
    else:
        env.exec(__file__, func=gen_clustering_plots, env='python_R', args=[run_settings])
        env.exec(__file__, func=process_good_clusterings, env='intel', args=[run_settings])

    # 2.2) Obtain 2D embeddings
    env.exec(__file__, func=calc_2d_embeddings, env='intel', args=[run_settings])

    # 3) Visualize the results
    env.exec(__file__, func=plot_clustering_quality, env='gnu', args=[run_settings])
    env.exec(__file__, func=plot_energies, env='gnu', args=[run_settings])
    env.exec(__file__, func=plot_diversity, env='python_R', args=[run_settings])
    env.exec(__file__, func=assemble_summary, env='gnu', args=[run_settings])

    """
    # These are a few more analysis modes that we have considered earlier

    # complete mode (keep all high-energy conformers for analysis, use clustering, show only cluster centers on diversity plots)
    run_settings = {
        'main_wd': './complete_ensemble_diversity',
        'mode': 'complete',
        'keep_unperspective': True,
        'keep_perspective': True,
        'use_refconformers': False,
        'plot_highenergy': True,
        'plot_perspective': True,
        'expand_clusters': False,
    }

    # low-energy mode (analyze only low-energy conformers, use clustering, show only cluster centers on diversity plots)
    run_settings = {
        'main_wd': './lowenergy_ensemble_diversity',
        'mode': 'lowenergy',
        'keep_unperspective': False,
        'keep_perspective': False,
        'use_refconformers': False,
        'plot_highenergy': False,
        'plot_perspective': False,
        'expand_clusters': False,
    }

    # perspective mode (keep all high-energy conformers for analysis, use refpoints instead of clustering, show only cluster centers on diversity plots)
    run_settings = {
        'main_wd': './perspective_ensemble_diversity',
        'mode': 'perspective',
        'keep_unperspective': True,
        'keep_perspective': True,
        'use_refconformers': True,
        'plot_highenergy': False,
        'plot_perspective': True,
        'expand_clusters': False,
    }
    """
