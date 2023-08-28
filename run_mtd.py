import os, json, time, glob, subprocess, signal, shutil

MTD_DIR = './mtd_temp'
TESTSET_JSON = 'testcases.json'
MTD_METHOD = 'mtd'
DF_FILENAME = f'{MTD_METHOD}_df.csv'
OUTPUT_DIR = f'{MTD_METHOD}_conformers'
MAX_TIME = 10 * 60 # Time in seconds

BOHR2A = 0.529177

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


def kill_xtb():
    # Get a list of process IDs for the xtb executable
    process_list = subprocess.check_output(["pidof", 'xtb']).split()

    # Send SIGTERM signal to each process
    for pid in process_list:
        os.kill(int(pid), signal.SIGTERM)

def kill_crest():
    # Get a list of process IDs for the crest executable
    process_list = subprocess.check_output(["pidof", 'crest']).split()

    # Send SIGTERM signal to each process
    for pid in process_list:
        os.kill(int(pid), signal.SIGTERM)

def fix_atom_symbol(symbol):
    return symbol[0].upper() + symbol[1:].lower()

def read_coord_file(filename, p):
    import numpy as np

    # Reads an XYZ file
    with open(filename, 'r') as f:
        lines = f.readlines()

    coords = []
    symbols = []
    for line in lines[1:]:
        if line.startswith('$end'):
            break
        tokens = line.split()
        x, y, z = float(tokens[0]), float(tokens[1]), float(tokens[2])
        coord = np.array([x, y, z])
        symbol = fix_atom_symbol(tokens[3])
        coords.append(coord)
        symbols.append(symbol)

    n_atoms = len(coords)
    xyz_matr = np.zeros((n_atoms, 3))
    for i, xyz in enumerate(coords):
        xyz_matr[i, :] = xyz * BOHR2A

    p.include_from_xyz(xyz_matr, f"Conformer {len(p)}")
    if len(p.atom_symbols[0]) > 0:
        assert p.atom_symbols == symbols, f"{repr(p.atom_symbols)} vs. {repr(symbols)}"
    else:
        p.atom_symbols = symbols


def gen_mtd(molname, sdf_name, p, charge=0):
    from chemscripts.geom import Molecule
    from chemscripts.utils import write_xyz

    # Set the input SDF file path and the output XYZ file path
    calc_dir = os.path.join(MTD_DIR, f'{molname}', 'crest')
    os.makedirs(calc_dir)
    start_xyzfile = os.path.join(calc_dir, 'start.xyz')

    # Convert SDF -> XYZ to start a short CREST calc
    ccmol = Molecule(sdf=sdf_name)
    xyzs, syms = ccmol.as_xyz()
    write_xyz(xyzs, syms, start_xyzfile)

    # Soon will kill this CREST process and take its MTD input
    crest_process = subprocess.Popen(f"./exec_dummycrest {calc_dir}", shell = True)

    # Prepare to spot the 'coord' file
    mtddir_name = os.path.join(calc_dir, "METADYN1")
    mtd_inpname = 'coord'
    mtdinput_fullname = os.path.join(mtddir_name, mtd_inpname)

    # Poll the process to check if it has created the directory
    while True:
        if not os.path.isfile(mtdinput_fullname):
            continue
        
        # When the file has been created, kill the process
        time.sleep(1) # Need to ensure that CREST finished writing the file
        os.kill(crest_process.pid, signal.SIGTERM)
        break

    # Make sure that xtb processes spawned by crest are not running
    kill_xtb()
    kill_crest()

    # Steal the 'coord' file
    EPIC_TIME = 100000.0 # Starting the longest md ever (will kill it when MAX_TIME is reached)
    mtdinput_contents = open(mtdinput_fullname, 'r').readlines()
    for i, line in enumerate(mtdinput_contents):
        if "time=" in line:
            mtdinput_contents[i] = "  time={:>8.2f}".format(EPIC_TIME)
    mtdinput_contents = ''.join(mtdinput_contents)

    # Setup 'calc_dir' for new calculation
    # shutil.rmtree(calc_dir) # Slow filesystems do not behave well from this...
    calc_dir = os.path.join(MTD_DIR, f'{molname}', 'xtb')
    os.mkdir(calc_dir)
    # A new name for xtb input
    mtdinput_fullname = os.path.join(calc_dir, mtd_inpname)
    with open(mtdinput_fullname, 'w') as f:
        f.write(mtdinput_contents)

    # Execute a new MTD calc outside of CREST
    subprocess.Popen(f"./exec_xtbmtd {calc_dir} {charge} {mtd_inpname}", shell = True)
    start_time = time.time()

    # Guess what happens when MAX_TIME has passed? 🔪
    while (time.time() - start_time) < MAX_TIME:
        time.sleep(0.05)
    kill_xtb()

    for snap_file in glob.glob(os.path.join(calc_dir, "scoord.*")):
        read_coord_file(snap_file, p)


def main():
    import pandas as pd
    import ringo
    from charges import CHARGES

    # Create output directoris if they don't exist
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    if os.path.exists(MTD_DIR):
        shutil.rmtree(MTD_DIR)
    os.makedirs(MTD_DIR)
    
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
        p = ringo.Confpool()

        # Must pass the molecular charge if it is not zero
        charge = 0
        if molname in CHARGES:
            charge = CHARGES[molname]

        with TimingContext(timer, molname):
            ntotal = gen_mtd(molname, sdf_name, p, charge=charge)

        p.generate_connectivity(0, mult=1.3, ignore_elements=['HCarbon'])
        print(f"Generating automorphisms of {molname}")
        niso = p.generate_isomorphisms()
        print(f"Found {niso} automorphisms of {molname}")
        ntotal = len(p)
        rmsd_status = p.rmsd_filter(0.2, mirror_match=True, print_status=True)
        print(f"[{molname}] Found {rmsd_status['DelCount']} duplicates")

        df['method'].append(MTD_METHOD)
        df['testcase'].append(molname)
        df['nconf'].append(ntotal)
        df['nunique'].append(len(p))
        df['time'].append(timer.time_elapsed)

        p.save(os.path.join(OUTPUT_DIR, f"{molname}_{MTD_METHOD}.xyz"))
        timer.finish_iter()

    df = pd.DataFrame(df)
    df.to_csv(DF_FILENAME, index=False)

    
if __name__ == "__main__":
    import environments as env
    env.exec(__file__, func=main, env='intel')
