import os, time

# MAX_TIME = 2 # s
MAX_TIME = 600 # s
MAX_SUCCESS = 10000
TESTSET_JSON = 'testcases.json'
NISO_JSON = 'niso_timings.json'


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


def main(env_name):
    RINGO_METHOD = env_name # Name of items in df
    DF_FILENAME = f'{RINGO_METHOD}_df.csv'
    OUTPUT_DIR = f'{RINGO_METHOD}_conformers'

    import sys, json
    import pandas as pd
    import ringo
    from chemscripts.geom import Molecule

    with open(NISO_JSON, "r") as f:
        niso_data = json.load(f)

    # Double-check that environment is running the right python executable
    print(f"The mode '{env_name}' is running on {sys.executable}")

    ringo.cleanup()
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

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
        # Should be testing on MacroModel testset only
        print(f"Starting with {molname}", flush=True)

        p = ringo.Confpool()
        start_mol = Molecule(sdf=sdf_name)

        rmsd_settings = False
        mol = ringo.Molecule(sdf=sdf_name)
        for item in niso_data:
            if item['mol'] == molname:
                if item['niso'] > 500:
                    biggest_frag_atoms = mol.get_biggest_ringfrag_atoms()
                    rmsd_settings = {
                            'isomorphisms': {
                                'ignore_elements': [node for node in start_mol.G.nodes if node not in biggest_frag_atoms],
                            },
                            'rmsd': {
                                'threshold': 0.2,
                                'mirror_match': True,
                            }
                        }
                else:
                    rmsd_settings = 'default'
        assert rmsd_settings is not None
        ringo.cleanup()

        with TimingContext(timer, molname):
            mol = ringo.Molecule(sdf=sdf_name)
            results = ringo.run_confsearch(mol, pool=p, rmsd_settings=rmsd_settings, timelimit=MAX_TIME, max_conformers=MAX_SUCCESS)

            # ntotal = gen_ringo(molname, sdf_name, p)
            ntotal = results['nsucc']

        df['method'].append(RINGO_METHOD)
        df['testcase'].append(molname)
        df['nconf'].append(ntotal)
        df['nunique'].append(len(p))
        df['time'].append(timer.time_elapsed)

        p.save(os.path.join(OUTPUT_DIR, f"{molname}_{RINGO_METHOD}.xyz"))
        timer.finish_iter()
    df = pd.DataFrame(df)
    df.to_csv(DF_FILENAME, index=False)
    ringo.cleanup()

if __name__ == "__main__":
    import environments as env
    env.exec(__file__, func=main, env='intel', args=('ringointel',))
