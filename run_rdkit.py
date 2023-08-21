import time

OUTPUT_DIR = 'rdkit_conformers'
TESTSET_JSON = 'testcases.json'
MAX_TIME = 10 * 60 # Time in seconds
# MAX_TIME = 5 # Time in seconds
RDKIT_METHOD = 'rdkit' # Name of items in df
DF_FILENAME = 'rdkit_df.csv'


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


def gen_rdkit(molname, input_file, p):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import numpy as np
    from chemscripts.geom import Molecule

    # Need to build RDKit molecule object from scratch
    ccmol = Molecule(sdf=input_file)
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
    
    # Generate conformers continuously for MAX_TIME seconds
    conformer_idx = 0
    start_time = time.time()
    while (time.time() - start_time) < MAX_TIME:
        # Embed molecule with random coordinates
        AllChem.EmbedMolecule(mol) # useRandomCoords=True

        # Write molecule as XYZ file
        geom = np.zeros((mol.GetNumAtoms(), 3))
        for i in range(mol.GetNumAtoms()):
            pos = mol.GetConformer().GetAtomPosition(i)
            geom[i, 0] = pos.x
            geom[i, 1] = pos.y
            geom[i, 2] = pos.z
        p.include_from_xyz(geom, f"Conformer {conformer_idx}")
        conformer_idx += 1
    p.atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

def main():
    import os, json
    import pandas as pd
    import ringo
    
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    with open(TESTSET_JSON, 'r') as f:
        input_data = json.load(f)
    
    INPUT_DIR = './testset_for_rdkit'
    df = {
        'method': [],
        'testcase': [],
        'nconf': [],
        'nunique': [],
        'time': [],
    }

    # RDKit requires charges, so files sdf are augmented with charges
    for molname, sdf_name in input_data.items():
        custom_input = os.path.isfile(os.path.join(INPUT_DIR, f'{molname}.sdf'))
        assert custom_input, f'Cannot find {custom_input}'

    timer = Timings()
    for molname, sdf_name in input_data.items():
        print(f"Starting with {molname}", flush=True)

        p = ringo.Confpool()

        with TimingContext(timer, molname):
            gen_rdkit(molname, os.path.join(INPUT_DIR, f'{molname}.sdf'), p)
        
        p.generate_connectivity(0, mult=1.3, ignore_elements=['HCarbon'])
        niso = p.generate_isomorphisms()
        ntotal = len(p)
        rmsd_status = p.rmsd_filter(0.2, mirror_match=True, print_status=False)
        print(f"[{molname}] Found {rmsd_status['DelCount']} duplicates")

        df['method'].append(RDKIT_METHOD)
        df['testcase'].append(molname)
        df['nconf'].append(ntotal)
        df['nunique'].append(len(p))
        df['time'].append(timer.time_elapsed)

        p.save(os.path.join(OUTPUT_DIR, f"{molname}_{RDKIT_METHOD}.xyz"))
        timer.finish_iter()
    df = pd.DataFrame(df)
    df.to_csv(DF_FILENAME, index=False)


if __name__ == "__main__":
    import environments as env
    # Ringo is needed only for Confpool
    # RINGO_FLAGS = f'-rmsd -eweak -mcrsmart -vara 5 -varb 1'
    # env.build_ringo('gnu', RINGO_FLAGS, __file__)

    # Execute 'main' function of this script in appropriate environment
    # main()
    env.exec(__file__, func=main, env='intelrdkit')
