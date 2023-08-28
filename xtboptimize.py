##############################
# WARNING: THIS SCRIPT IS NOT
# A PART OF THE BENCHMARK WORKFLOW!
# Use it to reproduce fails
# during GFN-FF optimizations
##############################
# module load intel-parallel-studio/2017 gcc/10.2.0

try:
    import glob, os, ntpath, sys, multiprocessing, json, subprocess, shutil
    import numpy as np
    import ringo
    from shutil import copy2
    import pandas as pd
    from chemscripts.geom import Molecule
    from chemscripts.utils import write_xyz
    import networkx as nx
    from charges import CHARGES
    from tqdm import tqdm
except:
    assert __name__ == "__main__"


# Number of threads to use for parallel segments of the workflow
NPROCS = 54

LOWER_DIR = './ringointel_conformers/'
XYZ_FILES = glob.glob(os.path.join(LOWER_DIR, '*.xyz'))
XTB_TEMP_DIR = './xtbpreopt_temp'
OPTIMIZED_DIR = './xtbpreoptimized_conformers'

TESTSET_JSON = 'testcases.json'
OPTSTATS_JSON = 'xtbpreoptimization_stats.json'
FAILSUMMARY_JSON = 'xtb_failsummary.json'

def parse_description(descr):
    descr_parts = descr.split('; ')
    parsed_descr = {
        'conf_name': descr_parts[0],
    }
    if len(descr_parts) > 1:
        parsed_descr['status'] = descr_parts[1].split(',')
        assert len(parsed_descr['status']) == 1
        parsed_descr['status'] = parsed_descr['status'][0]
    if len(descr_parts) == 3:
        parsed_descr['energy'] = float(descr_parts[2])
    return parsed_descr

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

def get_xtbopt_status(log_name):
    succ_found = False
    bad_found = False
    for line in open(log_name, 'r').readlines():
        if "GEOMETRY OPTIMIZATION CONVERGED AFTER" in line:
            succ_found = True
            break
        if "FAILED TO CONVERGE GEOMETRY OPTIMIZATION" in line:
            bad_found = True
            break
    assert succ_found or bad_found
    return succ_found

def xtboptimize_xyz(input_data):
    # Unpack the 'input_data' dict
    start_xyz = input_data['start_ensemble_xyz']
    initial_sdf = input_data['initial_sdf']
    res_xyz = input_data['optimized_xyz']
    molname = input_data['molname']
    assert os.path.isfile(start_xyz), f"Start XYZ named '{start_xyz}' not found"
    assert os.path.isfile(initial_sdf), f"Initial SDF named '{initial_sdf}' not found"

    index = input_data['index']
    max_index = input_data['max_index']
    print(f"Starting with job {index}/{max_index}", flush=True)
    
    initial_p = ringo.Confpool()
    initial_p.include_from_file(start_xyz)
    optimized_p = ringo.Confpool()

    initial_p['Index'] = lambda m: float(parse_description(m.descr)['conf_name'].split()[1])
    optimize_xyzs = []
    for conf_idx, m in enumerate(initial_p):
        dirname = "{}_{}".format(
            ntpath.basename(start_xyz).replace('.xyz', ''),
            str(int(m['Index']))
        )
        fulldir = os.path.join(XTB_TEMP_DIR, dirname)
        
        if os.path.isdir(fulldir):
            shutil.rmtree(fulldir)
        os.mkdir(fulldir)
        
        newxyz_name = os.path.join(fulldir, 'start.xyz')
        write_xyz(m.xyz, initial_p.atom_symbols, newxyz_name)
        optimize_xyzs.append(fulldir)

    if molname in CHARGES:
        charge = CHARGES[molname]
    else:
        charge = 0

    procs = []
    docalc = [dir for dir in optimize_xyzs]
    while len(docalc) > 0 or len(procs) > 0:
        for i in reversed(range(len(procs))):
            if not procs[i].poll() == None:
                del procs[i]
        while len(procs) < NPROCS and len(docalc) > 0:
            calcdir = docalc.pop()
            # print(f"Todo {len(docalc)}")
            procs.append(subprocess.Popen(f"./exec_xtbopt {calcdir} {charge}", shell = True))

    for conf_idx, calcdir in enumerate(optimize_xyzs):
        xtbopt_name = os.path.join(calcdir, 'xtbopt.xyz')
        opt_okay = os.path.exists(xtbopt_name)
        if opt_okay:    
            log_name = os.path.join(calcdir, 'log')
            opt_okay = get_xtbopt_status(log_name)
        
        if opt_okay:
            status = []
        else:
            status = ['optfail']

        if os.path.exists(xtbopt_name):
            optimized_p.include_from_file(xtbopt_name)
        else:
            xtblast_name = os.path.join(calcdir, 'xtblast.xyz')
            assert os.path.exists(xtblast_name), f'neither {xtbopt_name} nor {xtblast_name} were found'
            optimized_p.include_from_file(xtblast_name)
        older_status = parse_description(initial_p[conf_idx].descr)
        optimized_p[len(optimized_p) - 1].descr = f"{older_status['conf_name']}; {process_status(status)}"
        optimized_p.atom_symbols = initial_p.atom_symbols
        print(f'{conf_idx}/{len(optimize_xyzs)}', flush=True)

    for dirname in optimize_xyzs:
        shutil.rmtree(dirname)

    # Preprocessing of the starting molgraph
    ccmol = Molecule(sdf=initial_sdf)
    graph = ccmol.G
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
            current_status = descr_parts[1].split(',')
            current_status.append('topofail')
            m.descr = f'{descr_parts[0]}; {process_status(current_status)}'

    optimized_p.save(res_xyz)

    index = input_data['index']
    max_index = input_data['max_index']
    print(f"Job {index}/{max_index} has finished", flush=True)

def analyze_topofails(input_data):
    molname = input_data['molname']
    initial_sdf = input_data['initial_sdf']
    res_xyz = input_data['optimized_xyz']
    assert os.path.isfile(initial_sdf), f"Initial SDF named '{initial_sdf}' not found"
    assert os.path.isfile(res_xyz), f"Start XYZ named '{res_xyz}' not found"

    p = ringo.Confpool()
    p.include_from_file(res_xyz)

    ccmol = Molecule(sdf=initial_sdf)
    graph = ccmol.G
    older_bonds = set(edge for edge in graph.edges)
    
    fail_cases = {}
    for m in tqdm(p):
        if 'topofail' not in m.descr:
            continue
        
        p.generate_connectivity(m.idx, mult=1.3)
        optimized_graph = p.get_connectivity()
        optimized_bonds = set(edge for edge in optimized_graph.edges)
        optim_unique = []
        for bond in optimized_bonds:
            if bond not in older_bonds:
                optim_unique.append(bond)
        older_unique = []
        for bond in older_bonds:
            if bond not in optimized_bonds:
                older_unique.append(bond)
        
        assert older_bonds != optimized_bonds

        descr_parts = m.descr.split('; ')
        conf_name = descr_parts[0]

        fail_cases[conf_name] = {
            'broken': list(older_bonds - optimized_bonds),
            'extra': list(optimized_bonds - older_bonds),
        }
    
    index = input_data['index']
    max_index = input_data['max_index']
    print(f"Job {index}/{max_index} has finished", flush=True)
    return fail_cases

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
        p = ringo.Confpool()
        p.include_from_file(res_file)
        flag_counts = {}
        for m in p:
            m_flags = m.descr.split('; ')[1].split(',')
            # assert len(m_flags) == 1, f'{repr(m_flags)} in {res_file}'
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
            'initial_sdf': full_testset[molname],
            'start_ensemble_xyz': xyzfile,
            'optimized_xyz': os.path.join(OPTIMIZED_DIR, f'{clean_name}.xyz'),
            # For CREST/CRESTFAILED modifications
            'molname': molname,
            'method': method,
        })

    # Needed for printing the optimization progress
    for index, item in enumerate(input_data):
        item['index'] = index
        item['max_index'] = len(input_data)

    # start_xyz => optimized_xyz. Parallel optimization of all conformational ensembles
    for task in input_data:
        if os.path.isfile(task['optimized_xyz']):
            continue
        xtboptimize_xyz(task)

    # Check and summarize optimization status from all generated XYZs
    # summarize_flags(input_data, 'optimized_xyz')

    fail_summary = {}
    for task in input_data:
        fail_summary[task['molname']] = analyze_topofails(task)
    with open(FAILSUMMARY_JSON, 'w') as f:
        json.dump(fail_summary, f)


if __name__ == "__main__":
    import environments as env
    env.exec(__file__, func=main, env='intel')
