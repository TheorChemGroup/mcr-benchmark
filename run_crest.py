import os, json, time, subprocess, shutil

DO_GENCROSSING_FAILED = False # Then, after False repeat with True
# After iteration with DO_GENCROSSING_FAILED = False, list all failed calcs in GENERIC_CROSSING_FAILED:
GENERIC_CROSSING_FAILED = ['csd_FINWEE10', 'csd_MIWTER', 'csd_RULSUN', 'csd_YIVNOG', 'pdb_1NWX', 'pdb_2IYA', 'pdb_3M6G']

MTD_DIR = './crest_temp'
TESTSET_JSON = 'small_testcases.json'
if DO_GENCROSSING_FAILED:
    CREST_METHOD = 'crestfailed'
else:
    CREST_METHOD = 'crest'
DF_FILENAME = f'{CREST_METHOD}_df.csv'
OUTPUT_DIR = f'{CREST_METHOD}_conformers'
CPULOAD_JSON = f'cpuload_{CREST_METHOD}.json'
CPULOAD_DATA = {}


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


def gen_crest(molname, sdf_name, p, charge=0):
    import psutil
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

    if DO_GENCROSSING_FAILED:
        executable = 'exec_fullcrest_nocrossing'
    else:
        executable = 'exec_fullcrest'
    crest_process = subprocess.Popen(f"./{executable} {calc_dir} {charge}", shell = True)
    while crest_process.poll() is None:
        CPULOAD_DATA[molname].append(psutil.cpu_percent(interval=1) / 100 * 56)
        time.sleep(1)

    conformer_xyzname = os.path.join(calc_dir, 'crest_conformers.xyz')
    assert os.path.exists(conformer_xyzname), f'The result of CREST calc {conformer_xyzname} not found'
    p.include_from_file(conformer_xyzname)

def main():
    import pandas as pd
    import pyxyz
    from charges import CHARGES, CHARGES_MOLS

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
        if DO_GENCROSSING_FAILED:
            if molname not in GENERIC_CROSSING_FAILED:
                continue
        print(f"Starting with {molname}", flush=True)
        p = pyxyz.Confpool()
        CPULOAD_DATA[molname] = []

        # Must pass the molecular charge if it is not zero
        charge = 0
        if molname in CHARGES_MOLS:
            charge = CHARGES[molname]

        with TimingContext(timer, molname):
            ntotal = gen_crest(molname, sdf_name, p, charge=charge)

        p.generate_connectivity(0, mult=1.3, ignore_elements=['HCarbon'])
        print(f"Generating automorphisms of {molname}")
        niso = p.generate_isomorphisms()
        print(f"Found {niso} automorphisms of {molname}")
        ntotal = len(p)
        rmsd_status = p.rmsd_filter(0.2)
        print(f"[{molname}] Found {rmsd_status['DelCount']} duplicates")

        df['method'].append(CREST_METHOD)
        df['testcase'].append(molname)
        df['nconf'].append(ntotal)
        df['nunique'].append(len(p))
        df['time'].append(timer.time_elapsed)

        for m in p:
            m.descr = f"Conformer {len(m.idx)}"
        p.save(os.path.join(OUTPUT_DIR, f"{molname}_{CREST_METHOD}.xyz"))
        timer.finish_iter()

    df = pd.DataFrame(df)
    df.to_csv(DF_FILENAME, index=False)
    with open(CPULOAD_JSON, "w") as f:
        json.dump(CPULOAD_DATA, f)


if __name__ == "__main__":
    import environments as env
    env.exec(__file__, func=main, env='intel')
