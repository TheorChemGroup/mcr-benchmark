import os, json, time, subprocess, shutil

MM_DIR = './mmring_test'
TESTSET_JSON = 'small_testcases.json'
MMRING_METHOD = 'mmring'
DF_FILENAME = f'{MMRING_METHOD}_df.csv'
OUTPUT_DIR = f'{MMRING_METHOD}_conformers'
CPULOAD_JSON = f'cpuload_{MMRING_METHOD}.json'
CPULOAD_DATA = {}
SCHRODINGER_LOCATION = os.environ.get('SCHRODINGER')

IDX_TO_ELEMENT = {1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn', 51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg', 81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U', 93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es', 100: 'Fm', 101: 'Md', 102: 'No', 103: 'Lr', 104: 'Rf', 105: 'Db', 106: 'Sg', 107: 'Bh', 108: 'Hs', 109: 'Mt', 110: 'Ds', 111: 'Rg', 112: 'Cn', 113: 'Nh', 114: 'Fl', 115: 'Mc', 116: 'Lv', 117: 'Ts', 118: 'Og'}

################################################
# VEEEEERY IMPORTANT!
# 1) BACK UP MM_DIR
# 2) CHECK IF result_short = f'{project_name}-0-out.mae.gz' IS ACTUALLY THE RIGHT FILE
# 3) RESTART WITHOUT CALLING SCHRODINGER AND WITHOUT CLEANING UP!!
################################################

class TimingContext(object):
    def __init__(self, timings_inst, testcase):
        self.timings_inst = timings_inst
        self.testcase = testcase

    def __enter__(self):
        self.timings_inst.record_start(self.testcase)
        return self

    def __exit__(self, type, value, traceback):
        self.timings_inst.record_finish(self.testcase)


class Timings:
    def __init__(self):
        self.data = {}
        self.event_types = set()
        self.start_time = None
        self.time_elapsed = None
        self.recent_case = None

    def record_start(self, testcase):
        if testcase not in self.data:
            self.data[testcase] = {}
        assert self.start_time is None
        assert self.time_elapsed is None
        self.start_time = time.perf_counter()

    def record_finish(self, testcase):
        assert testcase in self.data
        assert self.start_time is not None
        assert self.time_elapsed is None

        self.time_elapsed = time.perf_counter() - self.start_time
        self.start_time = None
        self.recent_case = testcase

    def assign_recent(self, event_type, nitems=1):
        if nitems > 0:
            self.event_types.add(event_type)
            assert self.recent_case is not None
            if event_type not in self.data[self.recent_case]:
                self.data[self.recent_case][event_type] = []
            self.data[self.recent_case][event_type] += [self.time_elapsed/nitems] * nitems

    def finish_iter(self):
        self.time_elapsed = None
        self.recent_case = None


def runcmd(cmd):
    cmd_process = subprocess.Popen(cmd, shell = True)
    while cmd_process.poll() is None:
        time.sleep(1)

def convert_structures(start_name, result_name):
    if start_name.endswith('.sdf') and (result_name.endswith('.mae') or result_name.endswith('.mae.gz')):
        startext, resext = 'sdf', 'mae'
    elif (start_name.endswith('.mae') or start_name.endswith('.mae.gz')) and result_name.endswith('.sdf'):
        startext, resext = 'mae', 'sdf'
    else:
        raise RuntimeError(f'Unexpected file extensions for SDF <-> MAE conversion. start={start_name}, res={result_name}')
    runcmd(f"{SCHRODINGER_LOCATION}/utilities/sdconvert -i{startext} {start_name} -o{resext} {result_name}")
    assert os.path.exists(result_name)

def gen_mmring(molname, sdf_name, p):
    import psutil
    import numpy as np
    from openbabel import pybel

    # Set the input SDF file path and the output XYZ file path
    calc_dir = os.path.join(MM_DIR, f'{molname}')
    os.makedirs(calc_dir)
    start_sdffile = os.path.join(calc_dir, f'{molname}_start.sdf')
    shutil.copy2(sdf_name, start_sdffile)

    # Convert SDF -> MAE
    start_short = f'{molname}_start.mae'
    start_maefile = os.path.join(calc_dir, start_short)
    convert_structures(start_sdffile, start_maefile)

    # Create input for MacroModel
    project_name = molname
    # result_short = f'{project_name}-0-out.mae.gz'
    result_short = f'{project_name}-0-out-out.mae.gz'
    mm_process = subprocess.Popen(f'./exec_mmring {calc_dir} {start_short} {project_name}', shell = True)
    while mm_process.poll() is None:
        CPULOAD_DATA[molname].append(psutil.cpu_percent(interval=5) / 100 * 56)
        time.sleep(5)

    result_maefile = os.path.join(calc_dir, result_short)
    assert os.path.exists(result_maefile), f'Resulting file {result_maefile} not found'
    
    # Convert MAE -> SDF
    result_sdffile = os.path.join(calc_dir, f'{molname}_result.sdf')
    convert_structures(result_maefile, result_sdffile)

    # Read MOL2 into Confpool through OpenBabe
    for mol in pybel.readfile("sdf", result_sdffile):
        symbols = [IDX_TO_ELEMENT[atom.atomicnum] for atom in mol.atoms]
        xyz_matr = np.zeros((len(symbols), 3))
        for i, atom in enumerate(mol.atoms):
            xyz_matr[i, :] = atom.coords
        
        p.include_from_xyz(xyz_matr, f"Conformer {len(p)}")
        if len(p.atom_symbols[0]) > 0:
            assert p.atom_symbols == symbols, f"{repr(p.atom_symbols)} vs. {repr(symbols)}"
        else:
            p.atom_symbols = symbols


def main():
    import pandas as pd
    import numpy as np
    import pyxyz
    from charges import CHARGES, CHARGES_MOLS

    # Create output directoris if they don't exist
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    if os.path.exists(MM_DIR):
        shutil.rmtree(MM_DIR)
    os.makedirs(MM_DIR)
    
    with open(TESTSET_JSON, 'r') as f:
        input_data = json.load(f)

    df = {
        'method': [],
        'testcase': [],
        'nconf': [],
        'nunique': [],
        'time': [],
    }

    timer = Timings()
    for molname, sdf_name in input_data.items():
        print(f"Starting with {molname}", flush=True)
        p = pyxyz.Confpool()
        CPULOAD_DATA[molname] = []

        with TimingContext(timer, molname):
            ntotal = gen_mmring(molname, sdf_name, p)

        p.generate_connectivity(0, mult=1.3, ignore_elements=['HCarbon'])
        print(f"Generating automorphisms of {molname}")
        niso = p.generate_isomorphisms()
        print(f"Found {niso} automorphisms of {molname}")
        ntotal = len(p)
        rmsd_status = p.rmsd_filter(0.2)
        print(f"[{molname}] Found {rmsd_status['DelCount']} duplicates")

        df['method'].append(MMRING_METHOD)
        df['testcase'].append(molname)
        df['nconf'].append(ntotal)
        df['nunique'].append(len(p))
        df['time'].append(timer.time_elapsed)

        p.save(os.path.join(OUTPUT_DIR, f"{molname}_{MMRING_METHOD}.xyz"))
        timer.finish_iter()

    df = pd.DataFrame(df)
    df.to_csv(DF_FILENAME, index=False)
    with open(CPULOAD_JSON, "w") as f:
        json.dump(CPULOAD_DATA, f)


if __name__ == "__main__":
    import environments as env
    # main()
    env.exec(__file__, func=main, env='intelrdkit')
