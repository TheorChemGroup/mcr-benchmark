try:
    import glob, os, ntpath, sys, multiprocessing, json
    import numpy as np
    import pyxyz
    from shutil import copy2
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from chemscripts.geom import Molecule
    import ringo
    import networkx as nx

    # To supress RDKit warnings
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except:
    assert __name__ == "__main__"

XYZ_FILES = []
METHODS = ('rdkit', 'mtd', 'mmbasic', 'mmring', 'ringointel', 'crest', 'crestfailed')
for method in METHODS:
    conformers_dir = f'./{method}_conformers'
    assert os.path.isdir(conformers_dir), f'{conformers_dir} not found'
    for xyzfile in glob.glob(os.path.join(conformers_dir, '*.xyz')):
        XYZ_FILES.append(xyzfile)

# Number of threads to use for parallel segments of the workflow
NPROCS = 54

TESTSET_JSON = 'testcases.json'
OPTIMIZED_DIR = './optimized_conformers'
FILTERED_DIR = './filtered_conformers'
LOWER_DIR = './lower_conformers'
OPTSTATS_JSON = 'optimization_stats.json'
NISO_JSON = 'niso_timings.json'
FINAL_DF_NAME = 'benchmark_stats.csv'
RMSD_CUTOFF = 0.2
ENERGY_THRESHOLD = 15.0 # kcal/mol
PYXYZ_KWARGS = {
    'mirror_match': True,
    'print_status': False,
}
CPULOAD_JSON_NAMES = {
    'mmbasic': 'cpuload_mmbasic.json',
    'mmring': 'cpuload_mmring.json',
    'crest': 'crest_cpuload.json',
    'crestfailed': 'crestfailed_cpuload.json', # TODO Check if it's really used
}

def process_status(status):
    if len(status) == 0 or status == ['succ']:
        res = ['succ']
    else:
        okay = True
        for item in status:
            if item.endswith('fail'):
                okay = False
            elif item != 'succ':
                print(f"[WARNING] What is '{item}'?")
        if okay:
            res = ['succ'] + status
        else:
            if 'succ' in status:
                del status[status.index('succ')]
            res = status
    return ','.join(res)

def optimize_xyz(input_data):
    # Unpack the 'input_data' dict
    start_xyz = input_data['start_xyz']
    initial_sdf = input_data['initial_sdf']
    res_xyz = input_data['optimized_xyz']
    assert os.path.isfile(start_xyz), f"Start XYZ named '{start_xyz}' not found"
    assert os.path.isfile(initial_sdf), f"Initial SDF named '{initial_sdf}' not found"

    # Prepare
    ccmol = Molecule(sdf=initial_sdf)
    graph = ccmol.G
    m = Chem.Mol()
    mol = Chem.EditableMol(m)
    for atom in graph.nodes:
        new_atom = Chem.Atom(graph.nodes[atom]['symbol'])
        if 'chrg' in graph.nodes[atom]:
            new_atom.SetFormalCharge(graph.nodes[atom]['chrg'])
        new_idx = mol.AddAtom(new_atom)
        assert new_idx == atom
    for edge in graph.edges:
        mol.AddBond(*edge, Chem.BondType(graph[edge[0]][edge[1]]['type']))
    mol = mol.GetMol()
    Chem.SanitizeMol(mol)

    initial_p = pyxyz.Confpool()
    initial_p.include_from_file(start_xyz)
    optimized_p = pyxyz.Confpool()

    for m in initial_p:
        # For simplicity, the molecule object must have only one conformer
        mol.RemoveAllConformers()
        conf = Chem.Conformer(mol.GetNumAtoms())

        # Set the 3D coordinates for each atom in the conformer
        xyz = m.xyz
        for atom_idx in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(atom_idx, xyz[atom_idx])
        # Add the conformer to the molecule
        mol.AddConformer(conf)

        # Perform MMFF optimization on the molecule using the provided coordinates
        return_code = -1
        niter = 0
        while return_code != 0 and niter < 100:
            return_code = AllChem.MMFFOptimizeMolecule(mol, confId=0, maxIters=1000) # Use confId=0 to indicate the first (and only) conformer
            niter += 1

        if return_code == 0:
            status = []
        else:
            status = ['optfail']

        # Get the new geometry as np.array
        geom = np.zeros((mol.GetNumAtoms(), 3))
        for i in range(mol.GetNumAtoms()):
            pos = mol.GetConformer().GetAtomPosition(i)
            geom[i, 0] = pos.x
            geom[i, 1] = pos.y
            geom[i, 2] = pos.z

        optimized_p.include_from_xyz(geom, f'{m.descr}; {process_status(status)}')
        optimized_p.atom_symbols = initial_p.atom_symbols

    # Preprocessing of the starting molgraph
    older_bonds = set(edge for edge in graph.edges)
    
    for m in optimized_p:
        optimized_p.generate_connectivity(m.idx, mult=1.3)
        optimized_graph = optimized_p.get_connectivity()
        optimized_bonds = set(edge for edge in optimized_graph.edges)
        optim_unique = []
        for bond in optimized_bonds:
            if bond not in older_bonds:
                optim_unique.append(bond)
        older_unique = []
        for bond in older_bonds:
            if bond not in optimized_bonds:
                older_unique.append(bond)
        
        if older_bonds != optimized_bonds:
            descr_parts = m.descr.split('; ')
            current_status = descr_parts[1].split(',') # this is the format for future, when we can store several errors at once
            current_status.append('topofail')
            m.descr = f'{descr_parts[0]}; {process_status(current_status)}'

    optimized_p.save(res_xyz)

    index = input_data['index']
    max_index = input_data['max_index']
    print(f"Job {index}/{max_index} has finished", flush=True)

def perform_ringrmsd_filtering(p, niso_full):
    p.generate_connectivity(0, mult=1.3)

    graph = p.get_connectivity().copy()
    all_nodes = [i for i in graph.nodes]
    bridges = list(nx.bridges(graph))
    # Remove the bridges from the graph
    graph.remove_edges_from(bridges)
    # Get the connected components
    components_lists = [list(comp) for comp in nx.connected_components(graph)]
    rmsd_matrices = []
    for conn_component in components_lists:
        if len(conn_component) == 1:
            continue
        p.generate_connectivity(0, mult=1.3,
                                ignore_elements=[node for node in all_nodes 
                                                 if node not in conn_component])
        cur_graph = p.get_connectivity()
        assert cur_graph.number_of_nodes() == len(conn_component)
        niso = p.generate_isomorphisms()
        assert niso < niso_full, f'Local NIso {niso} is bigger than {niso_full}'
        matr = p.get_rmsd_matrix(**PYXYZ_KWARGS)
        rmsd_matrices.append(matr)

    first_shape = np.shape(rmsd_matrices[0])
    # Ensure the shapes are the same
    for matrix in rmsd_matrices[1:]:
        assert np.shape(matrix) == first_shape, "Matrices do not have the same shape."

    # Apply element-wise maximum across the matrices
    max_matrix = np.maximum.reduce(rmsd_matrices)
    n_deleted = p.rmsd_filter(RMSD_CUTOFF, rmsd_matrix=max_matrix, **PYXYZ_KWARGS)['DelCount']
    return n_deleted

def rmsd_filter(input_data):
    print(f"Processing {input_data['optimized_xyz']} to generate {input_data['filtered_xyz']}", flush=True)
    
    optimized_xyz = input_data['optimized_xyz']
    filtered_xyz = input_data['filtered_xyz']
    molname = input_data['molname']
    assert os.path.isfile(optimized_xyz)

    filtered_p = ringo.Confpool()
    filtered_p.include_from_file(optimized_xyz)

    # Descr has format 'Conformer 516; succ'
    descriptions_okay = True
    for m in filtered_p:
        if 'Conformer' not in m.descr:
            descriptions_okay = False
            break
    if not descriptions_okay:
        for m in filtered_p:
            status = m.descr.split('; ')[1]
            m.descr = f"Conformer {m.idx}; {status}"

    get_conformer_index = lambda m: float(m.descr.split()[1].replace(';', ''))
    filtered_p['Index'] = get_conformer_index
    # Filtering will be done only for successfully optimized geometries
    filtered_p.filter(lambda m: 'succ' in m.descr)

    # Choose a method of RMSD filtering
    # use_ring_rmsd = False
    # niso_full = None
    # with open(NISO_JSON, "r") as f:
    #     niso_data = json.load(f)
    # for item in niso_data:
    #     if item['mol'] == molname:
    #         niso_full = item['niso']
    #         if item['niso'] > 1000:
    #             use_ring_rmsd = True
    #         break
    # assert niso_full is not None
    
    # Perform RMSD filtering
    initial_size = len(filtered_p)
    # if not use_ring_rmsd:
    # The default way as it will disctiminate conformers in the most reliable way
    filtered_p.generate_connectivity(0, mult=1.3, ignore_elements=['HCarbon'])
    filtered_p.generate_isomorphisms()
    stats = filtered_p.rmsd_filter(RMSD_CUTOFF, **PYXYZ_KWARGS)
    n_deleted = stats['DelCount']
    # else:
    #     # If the number of automorphisms is too big, we use maximal RMSD of macrocycles
    #     n_deleted = perform_ringrmsd_filtering(filtered_p, niso_full)
    print(f"Deleted {n_deleted} for {molname}")
    assert initial_size - len(filtered_p) == n_deleted

    unique_conformers_idxs = [int(idx) for idx in filtered_p['Index']]

    resulting_p = ringo.Confpool()
    resulting_p.include_from_file(optimized_xyz)

    descriptions_okay = True
    for m in resulting_p:
        if 'Conformer' not in m.descr:
            descriptions_okay = False
            break
    if not descriptions_okay:
        for m in resulting_p:
            status = m.descr.split('; ')[1]
            m.descr = f"Conformer {m.idx}; {status}"

    for m in resulting_p:
        idx = get_conformer_index(m)
        descr_parts = m.descr.split('; ')
        current_status = descr_parts[1].split(',')
        assert len(current_status) == 1, f'{repr(current_status)} in {optimized_xyz}'
        if idx not in unique_conformers_idxs:
            if current_status[0] == 'succ':
                current_status.append('rmsdfail')
            m.descr = f'{descr_parts[0]}; {process_status(current_status)}'
    resulting_p.save(filtered_xyz)

def record_energies(input_data):
    filtered_xyz = input_data['filtered_xyz']
    lower_xyz = input_data['lower_xyz']
    initial_sdf = input_data['initial_sdf']
    assert os.path.isfile(filtered_xyz), f"XYZ named '{filtered_xyz}' not found"
    assert os.path.isfile(initial_sdf), f"Initial SDF named '{initial_sdf}' not found"

    print(f"{filtered_xyz} ==> {lower_xyz}. {input_data['index']}/{input_data['max_index']}")

    # Prepare RDKit object
    ccmol = Molecule(sdf=initial_sdf)
    graph = ccmol.G
    m = Chem.Mol()
    mol = Chem.EditableMol(m)
    for atom in graph.nodes:
        new_atom = Chem.Atom(graph.nodes[atom]['symbol'])
        if 'chrg' in graph.nodes[atom]:
            new_atom.SetFormalCharge(graph.nodes[atom]['chrg'])
        new_idx = mol.AddAtom(new_atom)
        assert new_idx == atom
    for edge in graph.edges:
        mol.AddBond(*edge, Chem.BondType(graph[edge[0]][edge[1]]['type']))
    mol = mol.GetMol()
    Chem.SanitizeMol(mol)

    # Will add energies to descriptions
    p = pyxyz.Confpool()
    p.include_from_file(filtered_xyz)

    for m in p:
        descr_parts = m.descr.split('; ')
        current_status = descr_parts[1].split(',')
        assert len(current_status) == 1, f'{repr(current_status)} in {filtered_xyz}'
        if 'fail' in current_status[0]:
            continue
        assert current_status[0] == 'succ'

        # For simplicity, the molecule object must have only one conformer
        mol.RemoveAllConformers()
        conf = Chem.Conformer(mol.GetNumAtoms())

        # Set the 3D coordinates for each atom in the conformer
        xyz = m.xyz
        for atom_idx in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(atom_idx, xyz[atom_idx])
        # Add the conformer to the molecule
        mol.AddConformer(conf)

        # Perform single-point energy calculation
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
        forcefield = AllChem.MMFFGetMoleculeForceField(mol, mmff_props)
        energy = forcefield.CalcEnergy()
        m.descr = m.descr + f'; {energy}'
    p.save(lower_xyz)

def lower_analysis(full_ensemble_data):
    print(f"Starting with {full_ensemble_data[0]['molname']}")
    total_p = pyxyz.Confpool()
    for single_method_data in full_ensemble_data:
        total_p.include_from_file(single_method_data['lower_xyz'])
    
    min_energy = None
    for m in total_p:
        descr_parts = m.descr.split('; ')
        if len(descr_parts) < 3:
            continue # Then, it's a failed structure
        
        energy = float(descr_parts[2])
        if min_energy is None or min_energy > energy:
            min_energy = energy
    assert min_energy is not None

    for single_method_data in full_ensemble_data:
        p = pyxyz.Confpool()
        p.include_from_file(single_method_data['lower_xyz'])
        for m in p:
            descr_parts = m.descr.split('; ')
            current_status = descr_parts[1].split(',')
            assert len(current_status) == 1, f"{repr(current_status)} in {single_method_data['lower_xyz']}"
            if 'fail' in current_status[0]:
                continue
            assert current_status[0] == 'succ'
            assert len(descr_parts) == 3

            energy = float(descr_parts[2])
            if energy - min_energy > ENERGY_THRESHOLD:
                current_status.append('lowerfail')
            m.descr = f'{descr_parts[0]}; {process_status(current_status)}; {energy}'
        p.save(single_method_data['lower_xyz']) # Overwriting


def process_chunk(input_data):
    function, xyz_chuck = input_data
    for xyz_file in xyz_chuck:
        function(xyz_file)

def parallelize_call(input_data, function):
    input_data_chunks = [[] for i in range(NPROCS)]
    next_chunk = 0
    for item in input_data:
        input_data_chunks[next_chunk].append(item)
        next_chunk += 1
        if next_chunk == NPROCS:
            next_chunk = 0
            
    with multiprocessing.Pool(processes=NPROCS) as pool:
        pool.map(process_chunk, [(function, chunk) for chunk in input_data_chunks])

def summarize_flags(input_data, key):
    flag_fulldata = {}
    for cur_idx, item in enumerate(input_data):
        print(f"Progress {cur_idx}/{len(input_data)}")
        res_file = item[key]
        p = pyxyz.Confpool()
        p.include_from_file(res_file)
        flag_counts = {}
        for m in p:
            m_flags = m.descr.split('; ')[1].split(',')
            assert len(m_flags) == 1, f'{repr(m_flags)} in {res_file}'
            for flag in m_flags:
                if flag not in flag_counts:
                    flag_counts[flag] = 1
                else:
                    flag_counts[flag] += 1
        flag_fulldata[res_file] = flag_counts
    with open(OPTSTATS_JSON, "w") as f:
        json.dump(flag_fulldata, f, indent=4)

def main():
    # Import full testset to get 'initial_sdf'
    with open(TESTSET_JSON, 'r') as f:
        full_testset = json.load(f)

    # General layout of input and output files
    input_data = []
    for index, xyzfile in enumerate(XYZ_FILES):
        clean_name = ntpath.basename(xyzfile).replace('.xyz', '') # csd_AAGGAG10_ringointel
        molname = '_'.join(clean_name.split('_')[:-1]) # csd_AAGGAG10
        method = clean_name.split('_')[-1] # ringointel
        input_data.append({
            # Key properties
            'start_xyz': xyzfile,
            'initial_sdf': full_testset[molname],
            'optimized_xyz': os.path.join(OPTIMIZED_DIR, f'{clean_name}.xyz'),
            'filtered_xyz': os.path.join(FILTERED_DIR, f'{clean_name}.xyz'),
            'lower_xyz': os.path.join(LOWER_DIR, f'{clean_name}.xyz'),
            # For CREST/CRESTFAILED modifications
            'molname': molname,
            'method': method,
        })
    
    if 'crestfailed' in METHODS:
        # There was a hack with crest/crestfailed. Fixing it here:
        all_crestfailed_removed = False
        while not all_crestfailed_removed:
            crestfailed_index = None
            crestfailed_item = None
            for index, item in enumerate(input_data):
                if item['method'] == 'crestfailed':
                    crestfailed_index = index
                    crestfailed_item = item
                    break
            if crestfailed_index is None:
                all_crestfailed_removed = True
                continue # Could have just 'break'
            
            # If crestfailed item was found, find the corresponding crest item
            crest_index = None
            crest_item = None
            for index, item in enumerate(input_data):
                if item['method'] == 'crest' and item['molname'] == crestfailed_item['molname']:
                    crest_index = index
                    crest_item = item
            assert crest_index is not None
            assert crestfailed_item['initial_sdf'] == crest_item['initial_sdf']
            crest_item['start_xyz'] = crestfailed_item['start_xyz']
            del input_data[crestfailed_index]

    # Needed for printing the optimization progress
    for index, item in enumerate(input_data):
        item['index'] = index
        item['max_index'] = len(input_data)

    # start_xyz => optimized_xyz. Parallel optimization of all conformational ensembles
    parallelize_call(input_data, optimize_xyz)

    # optimized_xyz => filtered_xyz. Parallel RMSD filtering of all conformational ensembles:
    parallelize_call(input_data, rmsd_filter)
    
    # Record energies of each conformer
    parallelize_call(input_data, record_energies)
    molnames_set = set(item['molname'] for item in input_data)
    full_ensembles_data = [
        [item for item in input_data
            if item['molname'] == molname] # Group items with the same molnames
        for molname in molnames_set
    ]
    parallelize_call(full_ensembles_data, lower_analysis)

    # Check and summarize optimization status from all generated XYZs
    summarize_flags(input_data, 'optimized_xyz')
    summarize_flags(input_data, 'lower_xyz')
    
    # Combine all timings in a single df
    df_parts = []
    for method in METHODS:
        df_name = f'{method}_df.csv'
        df_parts.append(pd.read_csv(df_name))
    df = pd.concat(df_parts, ignore_index=True)

    # Include cpuload if multithreading was used with this method
    methods = df['method'].unique()
    df['thread_avg'] = np.nan
    for method in methods:
        if method not in CPULOAD_JSON_NAMES:
            df.loc[df['method'] == method, 'thread_avg'] = 1.0
            continue
        
        with open(CPULOAD_JSON_NAMES[method], 'r') as f:
            cpuload_full = json.load(f)
        
        # cpuload_full needs to map 'molname' => 'average CPU load for this molecule'
        cpuload_full = {molname: sum(loadlist)/len(loadlist) for molname, loadlist in cpuload_full.items()}
        
        # Record thread_avg for each molecule
        for testcase, thread_avg in cpuload_full.items():
            df.loc[(df['method'] == method) & (df['testcase'] == testcase), 'thread_avg'] = thread_avg

    # === This replaces a crest calc if a crestfailed calc exists for the same molname ===
    # Create a mask to identify 'crest' rows with corresponding 'crestfailed' rows
    mask = df['method'] == 'crestfailed'
    # Extract unique 'testcase' values for which there are 'crestfailed' rows
    testcases_with_crestfailed = df.loc[mask, 'testcase'].unique()
    # Filter rows where 'method' is 'crest' and 'testcase' is not in the 'testcases_with_crestfailed' list
    df = df[~((df['method'] == 'crest') & (df['testcase'].isin(testcases_with_crestfailed)))].reset_index(drop=True)
    df['method'] = df['method'].replace('crestfailed', 'crest')

    # Now we compute these four statistics
    df['n_duplicates'] = np.nan
    df['n_failed_opts'] = np.nan
    df['n_failed_topo'] = np.nan
    df['n_higher_unique'] = np.nan
    df['n_lower_unique'] = np.nan
    for idx, item in enumerate(input_data):
        lower_xyz_file = item['lower_xyz']
        print(f"Processing {lower_xyz_file}: {idx}/{len(input_data)}")
        method = item['method']
        testcase = item['molname']

        n_duplicates = 0
        n_failed_opts = 0
        n_failed_topo = 0
        n_higher_unique = 0
        n_lower_unique = 0

        p = ringo.Confpool()
        p.include_from_file(lower_xyz_file)
        for m in p:
            descr_parts = m.descr.split('; ')
            current_status = descr_parts[1].split(',')
            # assert len(current_status) == 1, f"WHAT??? {repr(current_status)}"
            current_status = current_status[0]
            if current_status == 'optfail':
                n_failed_opts += 1
            elif current_status == 'topofail':
                n_failed_topo += 1
            elif current_status == 'rmsdfail':
                n_duplicates += 1
            elif current_status == 'lowerfail':
                n_higher_unique += 1
            elif current_status == 'succ':
                n_lower_unique += 1
            else:
                raise Exception(f"{current_status} - what is this?")

        df.loc[(df['method'] == method) & (df['testcase'] == testcase), 'n_duplicates'] = n_duplicates
        df.loc[(df['method'] == method) & (df['testcase'] == testcase), 'n_failed_opts'] = n_failed_opts
        df.loc[(df['method'] == method) & (df['testcase'] == testcase), 'n_failed_topo'] = n_failed_topo
        df.loc[(df['method'] == method) & (df['testcase'] == testcase), 'n_higher_unique'] = n_higher_unique
        df.loc[(df['method'] == method) & (df['testcase'] == testcase), 'n_lower_unique'] = n_lower_unique

    # == Final strokes on the dataframe ==
    # 1) Fix types of the new columns
    for column in ('n_duplicates', 'n_failed_opts', 'n_failed_topo', 'n_higher_unique', 'n_lower_unique'):
        df[column] = df[column].astype(int)
    # 2) 'nconf' 'nunique' are not so informative by now
    df.rename(columns={
        'nconf': 'n_total_generated',
        'nunique': 'n_unique_generated',
    }, inplace=True)
    # 3) Check that we're good
    if df.isna().any().any():
        print("DataFrame contains NaN values!!!")
    
    df.to_csv(FINAL_DF_NAME, index=False)


if __name__ == "__main__":
    import environments as env
    env.exec(__file__, func=main, env='intel')
