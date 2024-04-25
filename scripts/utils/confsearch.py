import os
import glob
import time
import json
import ntpath
import random
import subprocess
import signal
import psutil
import itertools
import shutil
import multiprocessing
from typing import Dict, Tuple, Any, List

import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism

from rdkit import Chem
from rdkit.Chem import AllChem
from chemscripts import geom as ccgeom
from chemscripts.utils import is_float, is_int, H2KC


BOHR2A = 0.529177

# Total basics

GJF_TEMPL = lambda calc: f'./inpfile_templates/{calc}_gaussian.gjf'
GAUSSIAN_PARAMETERS = {
    'level': 'rpm3',
    'nproc': 6,
}

def fix_element_str(sym: str) -> str:
    return sym[0].upper() + sym[1:].lower()

def assert_path(path):
    assert os.path.exists(path), f"Cannot find the file/dir '{path}'"
    return path
    
def include_if_exists(path, item, **keys):
    if os.path.exists(path):
        item.include_element(path, **keys)
        return True
    else:
        return False

def load_if_exists(item, **keys):
    return include_if_exists(path=item.get_path(**keys), item=item, **keys)

def assertive_include(item):
    return item.include_element(assert_path(item.get_path()))


# Core confsearch stuff starts here:

def rdkitmol_from_sdf(sdf_name: str, sanitize=True, load_xyz=False):
    ccmol = ccgeom.Molecule(sdf=sdf_name)
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
    if sanitize:
        Chem.SanitizeMol(mol)
    
    if load_xyz:
        rdkit_add_coordinates(mol, [
            graph.nodes[node]['xyz']
            for node in range(graph.number_of_nodes())
        ])
    return mol

def rdkit_add_coordinates(mol, xyz):
    mol.RemoveAllConformers()
    conf = Chem.Conformer(mol.GetNumAtoms())

    # Set the 3D coordinates for each atom in the conformer
    for atom_idx in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(atom_idx, xyz[atom_idx])
    # Add the conformer to the molecule
    mol.AddConformer(conf)

def rdkitmol_optimize(mol):
    niter = 0
    return_code = None
    while return_code != 0 and niter < 1000:
        return_code = AllChem.MMFFOptimizeMolecule(mol, confId=0, maxIters=1000) # Use confId=0 to indicate the first (and only) conformer
        niter += 1
    if return_code != 0:
        raise RuntimeError("Optimization failed")

def rdkit_singlepoint(mol):
    mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
    forcefield = AllChem.MMFFGetMoleculeForceField(mol, mmff_props)
    return forcefield.CalcEnergy()

def rdkit_as_xyz(mol):
    xyz = np.zeros((mol.GetNumAtoms(), 3))
    for i in range(mol.GetNumAtoms()):
        pos = mol.GetConformer().GetAtomPosition(i)
        xyz[i, 0] = pos.x
        xyz[i, 1] = pos.y
        xyz[i, 2] = pos.z
    
    symbols = []
    for i, atom in enumerate(mol.GetAtoms()):
        assert i == atom.GetIdx()
        symbols.append(atom.GetSymbol())
    return xyz, symbols

def RDKIT_BONDTYPE_MAP(btype):
    if btype < 10:
        return btype
    elif btype == 12:
        return 2
    else:
        print(f"[WARNING!!!!] Unknown bond type '{btype}'")
        return btype
        # raise RuntimeError(f"Unknown bond type '{btype}'")

def rdkit_save_sdf(mol, sdf_name, template_sdf=None):
    geom = np.zeros((mol.GetNumAtoms(), 3))
    for i in range(mol.GetNumAtoms()):
        pos = mol.GetConformer().GetAtomPosition(i)
        geom[i, 0] = pos.x
        geom[i, 1] = pos.y
        geom[i, 2] = pos.z

    if template_sdf is None:
        testmol = ccgeom.Molecule(shutup=True)
        testmol.G = nx.Graph()
        for atom in mol.GetAtoms():
            testmol.G.add_node(atom.GetIdx(),
                symbol=atom.GetSymbol(),
                xyz=geom[atom.GetIdx()],
                chrg=atom.GetFormalCharge()
            )
        for bond in mol.GetBonds():
            testmol.G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), type=RDKIT_BONDTYPE_MAP(int(bond.GetBondType())))
    else:
        testmol = ccgeom.Molecule(sdf=template_sdf)
        for node, keys in testmol.G.nodes(data=True):
            keys['xyz'] = geom[node]
            
    testmol.save_sdf(sdf_name)

def same_element(n1_attrib, n2_attrib):
    return n1_attrib['symbol'] == n2_attrib['symbol']

def same_bondtype(e1_attrib, e2_attrib):
    return e1_attrib['type'] == e2_attrib['type']

def get_amide_bonds(sdf_name):
    mol = ccgeom.Molecule(sdf=sdf_name)
    graph = mol.G

    amidegroup = nx.Graph()
    amidegroup.add_node(0, symbol='C')
    amidegroup.add_node(1, symbol='N')
    amidegroup.add_node(2, symbol='O')
    amidegroup.add_node(3, symbol='H')
    amidegroup.add_node(4, symbol='C')
    amidegroup.add_edge(0, 1, type=1)
    amidegroup.add_edge(0, 2, type=2)
    amidegroup.add_edge(3, 1, type=1)
    amidegroup.add_edge(4, 1, type=1)

    # Initialize the subgraph isomorphism matcher
    matcher = isomorphism.GraphMatcher(graph, amidegroup, node_match=same_element, edge_match=same_bondtype)
    
    # Find all matches of the subgraph in the larger graph
    amide_bonds = []
    full_amide_bonds = []
    for match in matcher.subgraph_isomorphisms_iter():
        rev_match = {value: key for key, value in match.items()}
        nitrogen_idx = rev_match[1]
        carbon_idx = rev_match[0]
        oxygen_idx = rev_match[2]
        hydrogen_idx = rev_match[3]
        amide_bonds.append((carbon_idx+1, nitrogen_idx+1))
        full_amide_bonds.append((oxygen_idx+1, carbon_idx+1, nitrogen_idx+1, hydrogen_idx+1))
    return amide_bonds, full_amide_bonds

def check_dihedral_limits(dih_value, lower_limit, upper_limit):
    upd_dih = dih_value
    while upd_dih < lower_limit:
        upd_dih += 360.0
    while upd_dih > upper_limit:
        upd_dih -= 360.0
    return (lower_limit < upd_dih) and (upd_dih < upper_limit), f"Limits=[{lower_limit}, {upper_limit}] Old = {dih_value} New = {upd_dih}"

def sdf_to_confpool(sdf_name, calc_automorphisms=True, description=''):
    from ringo import Confpool

    mol = ccgeom.Molecule(sdf=sdf_name)
    p = Confpool()
    xyz, sym = mol.as_xyz()
    p.include_from_xyz(xyz, description)
    p.atom_symbols = sym
    if calc_automorphisms:
        p.generate_connectivity(0, mult=1.3, ignore_elements=['HCarbon'])
        p.generate_isomorphisms()
    return p

def sdf_to_xyz(sdf_path: str, xyz_path: str, description='') -> str:
    p = sdf_to_confpool(sdf_path, description=description, calc_automorphisms=False)
    p.save_xyz(xyz_path)
    return xyz_path

def print_ringo_warnings(stage_name, molpath):
    import ringo

    warnings = ringo.get_status_feed(important_only=True)
    if len(warnings) > 0:
        print('---------------------')
        print(f"Please study these {stage_name} warnings carefully (mol={molpath}):")
        for item in warnings:
            print("* " + item['message'])
            if 'atoms' in item:
                print("Atoms = "+ repr(item['atoms']))
        print('---------------------')


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


class TimingContext:
    def __init__(self, timings_inst, testcase):
        self.timings_inst = timings_inst
        self.testcase = testcase

    def __enter__(self):
        self.timings_inst.record_start(self.testcase)
        return self

    def __exit__(self, type, value, traceback):
        self.timings_inst.record_finish(self.testcase)


class ConformerInfo:
    NECESSARY_ATTRS = ['name', 'statusflags']
    ALL_ALLOWED_ATTRS = [*NECESSARY_ATTRS, 'method', 'energy', 'relenergy', 'time', 'generated']

    def __init__(self,
            description: str=None,
            data: Dict[str, Any]=None
        ) -> None:
        self.data = None
        assert (data is not None) ^ (description is not None)
        if description is not None:
            self._from_json(description)
        if data is not None:
            self._from_dict(data)
    
    def _from_dict(self, passed_dict: Dict[str, Any]):
        assert self.data is None, f"Attempted overwrite of ConformerInfo object. Old data={self.data}"
        self.data = passed_dict
        self._validate_data()

    def _from_json(self, json_str: str) -> None:
        assert self.data is None, f"Attempted overwrite of ConformerInfo object. Old data={self.data}"
        self.data = json.loads(json_str)
        self._validate_data()
    
    def _validate_data(self):
        assert all(
            key in self.data
            for key in self.NECESSARY_ATTRS
        ), f"Some necessary keys were not found: {self.data}"
        assert all(
            key in self.ALL_ALLOWED_ATTRS
            for key in self.data.keys()
        ), f"Some keys were not recognized: {self.data}"
    
    @staticmethod
    def _lazy_validate_json(text: str) -> bool:
        return text.startswith('{') and text.endswith('}')

    @staticmethod
    def _read_raw_description(raw_descr: str, conf_count: int) -> Tuple["ConformerInfo", str]:
        # Types of descriptions:
        # 1) "Conformer %d" (conf_index)
        # 1') "Experimental conformer" (exp_conformer)
        # 2) "        -14.35514378" (crestOld)
        # 3) "No post-optimization.;%f" (mcr_timed)
        
        descr_parts = raw_descr.split()
        if len(descr_parts) == 1:
            assert is_float(descr_parts[0]), f"Cannot parse '{raw_descr}'"
            conformer_data = {
                'name': f'Conformer {conf_count}',
                'statusflags': []
            }
            descr_type = 'crestOld'
        elif len(descr_parts) == 2 and descr_parts[0] == 'Conformer':
            assert is_int(descr_parts[1]), f"Cannot parse '{raw_descr}'"
            conformer_data = {
                'name': raw_descr,
                'statusflags': []
            }
            descr_type = 'conf_index'
        elif len(descr_parts) == 2 and raw_descr == 'Experimental conformer':
            conformer_data = {
                'name': raw_descr,
                'statusflags': []
            }
            descr_type = 'exp_conformer'
        elif raw_descr.startswith('No post-optimization.;'):
            semicolon_parts = raw_descr.split(';')
            assert len(semicolon_parts) == 2, f"Cannot parse '{raw_descr}'"
            gentime_str = semicolon_parts[1]
            assert is_float(gentime_str), f"Cannot parse '{raw_descr}'"
            conformer_data = {
                'name': f'Conformer {conf_count}',
                'time': float(gentime_str),
                'statusflags': []
            }
            descr_type = 'mcr_timed'
        else:
            raise RuntimeError(f"Cannot parse '{raw_descr}'")

        return ConformerInfo(data=conformer_data), descr_type

    @staticmethod
    def standardize_confpool_descriptions(p) -> None:
        if all(
            ConformerInfo._lazy_validate_json(m.descr)
            for m in p
        ):
            return None

        common_descr_type = None
        for conf_count, m in enumerate(p):
            old_descr = m.descr
            status, descr_type = ConformerInfo._read_raw_description(old_descr, conf_count)

            if descr_type == 'exp_conformer':
                assert len(p) == 1, f"XYZ file with experimental conformer is allowed to contain only 1 structure. Got {len(p)}"

            if common_descr_type is not None:
                assert common_descr_type == descr_type
            else:
                common_descr_type = descr_type

            m.descr = status.as_str()

    def _include_status(self, status_update: str) -> None:
        if len(self.data['statusflags']) == 0:
            self.data['statusflags'].append(status_update)
        elif self.data['statusflags'] == ['succ'] and status_update == 'succ':
            pass
        elif len(self.data['statusflags']) != 0 and status_update != 'succ':
            assert status_update.endswith('fail'), f"Unknown status '{status_update}'"
            if 'succ' in self.data['statusflags']:
                self.data['statusflags'].remove('succ')
            self.data['statusflags'].append(status_update)
        else:
            raise RuntimeError(f"Cannot add status '{status_update}' to existing flags '{self.data['statusflags']}'")
        
    def _include_energy(self, energy: float) -> None:
        # assert 'energy' not in self.data
        assert isinstance(energy, float)
        self.data['energy'] = energy
        
    def _include_relenergy(self, relenergy: float) -> None:
        # assert 'relenergy' not in self.data
        assert isinstance(relenergy, float)
        self.data['relenergy'] = relenergy

    @staticmethod
    def updated_status(
        old_descr: str,
        update: str=None,
        energy: float=None,
        relenergy: float=None,
    ) -> str:
        obj = ConformerInfo(description=old_descr)
        if update is not None:
            obj._include_status(update)
        if energy is not None:
            obj._include_energy(energy)
        if relenergy is not None:
            obj._include_relenergy(relenergy)
        return obj.as_str()

    def as_str(self) -> str:
        return json.dumps(self.data)

    @property
    def is_failed(self) -> bool:
        assert len(self.data['statusflags']) > 0
        return any(
            flag.endswith('fail')
            for flag in self.data['statusflags']
        )

    @property
    def is_successful(self) -> bool:
        result = not self.is_failed
        if result:
            assert 'succ' in self.data['statusflags']
        return result

    @property
    def energy(self) -> float:
        return self.data['energy']

    @property
    def relenergy(self) -> float:
        return self.data['relenergy']

    @property
    def index(self) -> int:
        name_parts = self.data['name'].split()
        assert name_parts[0] == 'Conformer'
        return int(name_parts[1])

    @property
    def status_flag(self) -> str:
        # assert len(self.data['statusflags']) == 1
        return self.data['statusflags'][0]


def process_conformer_status(status_original):
    status = [x for x in status_original]
    if len(status) == 0 or status == ['succ']:
        res = ['succ']
    else:
        okay = True
        for item in status:
            if item.endswith('fail'):
                okay = False
            elif item != 'succ':
                print(f"[WARNING] What is '{item}'?")
        if okay and 'succ' not in status:
            res = ['succ'] + status
        if not okay and 'succ' in status:
            del status[status.index('succ')]
        res = status
    return ','.join(res)

class TopologyChecker:
    def __init__(self, correct_graph: nx.Graph):
        self.correct_graph = correct_graph
        self.correct_bonds = set(edge for edge in correct_graph.edges)
    
    def same_topology(self, check_graph: nx.Graph):
        check_bonds = set(edge for edge in check_graph.edges)
        optim_unique = []
        for bond in check_bonds:
            if bond not in self.correct_bonds:
                optim_unique.append(bond)
        older_unique = []
        for bond in self.correct_bonds:
            if bond not in check_bonds:
                older_unique.append(bond)
        return self.correct_bonds == check_bonds, optim_unique, older_unique


def get_conformer_description(name: str, status, energy: float=None):
    if isinstance(status, list):
        status = process_conformer_status(status)
    assert isinstance(status, str), f"Is not a status '{repr(status)}'"

    descr_dict = {
        'status': status,
        'name': name
    }
    if energy is not None:
        descr_dict['energy'] = energy
    
    new_descr = json.dumps(descr_dict)
    new_descr = new_descr.strip()
    assert '\n' not in new_descr, f"Description {repr(new_descr)} is invalid"
    return new_descr


def parse_conformer_description(description):
    data = json.loads(description)
    data['status'] = data['status'].split(',')
    return data

def json_from_file(filename: str) -> Dict:
    with open(filename, 'r') as f:
        result = json.load(f)
    return result

def sdf_to_smiles(sdf_name: str) -> str:
    # Need to build RDKit molecule object from scratch
    ccmol = ccgeom.Molecule(sdf=sdf_name)
    graph = ccmol.G

    with open(sdf_name, 'r') as f:
        sdf_lines = f.readlines()
    tmp_sdf = f'/tmp/mol-{random.randint(0, 100000)}.sdf'
    with open(tmp_sdf, 'w') as f:
        f.write(''.join([
            line
            for line in sdf_lines
            if 'CHG' not in line
        ]))

    supplier = Chem.SDMolSupplier(tmp_sdf, sanitize=False, removeHs=False)
    mol = supplier[0]
    assert mol is not None, \
        f"RDKit fails to parse the molecule '{sdf_name}'"

    for atom in graph.nodes:
        if 'chrg' in graph.nodes[atom]:
            mol.GetAtomWithIdx(atom).SetFormalCharge(graph.nodes[atom]['chrg'])

    smiles = Chem.MolToSmiles(mol)
    os.remove(tmp_sdf)
    return smiles


def pidof(pgname):
    pids = []
    for proc in psutil.process_iter(['name', 'cmdline']):
        # search for matches in the process name and cmdline
        if proc.info['name'] == pgname or \
                proc.info['cmdline'] and proc.info['cmdline'][0] == pgname:
            pids.append(str(proc.pid))
    return pids

def kill_xtb():
    # Get a list of process IDs for the xtb executable
    process_list = pidof('xtb')

    # Send SIGTERM signal to each process
    for pid in process_list:
        os.kill(int(pid), signal.SIGTERM)

def kill_crest():
    # Get a list of process IDs for the crest executable
    process_list = pidof('crest')

    # Send SIGTERM signal to each process
    for pid in process_list:
        os.kill(int(pid), signal.SIGTERM)

def fix_atom_symbol(symbol):
    return symbol[0].upper() + symbol[1:].lower()

def read_coord_file(filename, p):
    # Reads an XYZ file
    with open(filename, 'r') as f:
        lines = f.readlines()

    coords = []
    symbols = []
    for line in lines[1:]:
        if line.startswith('$eht'):
            continue
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


def gen_mtd(sdf_name, temp_dir, p, timelimit, charge=0):
    from chemscripts.utils import write_xyz

    # Set the input SDF file path and the output XYZ file path
    calc_dir = os.path.join(temp_dir, 'crest')
    os.makedirs(calc_dir)
    start_xyzfile = os.path.join(calc_dir, 'start.xyz')

    # Convert SDF -> XYZ to start a short CREST calc
    ccmol = ccgeom.Molecule(sdf=sdf_name)
    xyzs, syms = ccmol.as_xyz()
    write_xyz(xyzs, syms, start_xyzfile)

    # Soon will kill this CREST process and take its MTD input
    crest_command = f"exec_dummycrest.sh {calc_dir}"
    ic(crest_command)
    crest_process = subprocess.Popen(crest_command, shell = True)

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
    calc_dir = os.path.join(temp_dir, 'xtb')
    os.mkdir(calc_dir)
    # A new name for xtb input
    mtdinput_fullname = os.path.join(calc_dir, mtd_inpname)
    with open(mtdinput_fullname, 'w') as f:
        f.write(mtdinput_contents)

    # Execute a new MTD calc outside of CREST
    subprocess.Popen(f"exec_xtbmtd.sh {calc_dir} {charge} {mtd_inpname}", shell = True)
    start_time = time.time()

    # Guess what happens when MAX_TIME has passed? 🔪
    while (time.time() - start_time) < timelimit:
        time.sleep(0.05)
    kill_xtb()

    for snap_file in glob.glob(os.path.join(calc_dir, "scoord.*")):
        read_coord_file(snap_file, p)

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

def get_energy_from_xtbdescr(descr: str) -> float:
    descr_parts = descr.split()
    assert descr_parts[0] == 'energy:', f"Invalid descr float: {descr}"
    assert is_float(descr_parts[1]), f"Invalid descr float: {descr}"
    return float(descr_parts[1])


def optimize_mmff_and_geomcheck(
    initial_sdf: str,
    res_xyz: str,
    start_xyz: str=None,
    initial_p: str=None,
) -> None:
    assert (initial_p is not None) ^ (start_xyz is not None)
    from ringo import Confpool

    ccmol = ccgeom.Molecule(sdf=initial_sdf)
    graph = ccmol.G
    mol = rdkitmol_from_sdf(initial_sdf)

    if start_xyz is not None:
        initial_p = Confpool()
        initial_p.include_from_file(start_xyz)

    optimized_p = Confpool()

    for m in initial_p:
        rdkit_add_coordinates(mol, m.xyz)

        # Perform MMFF optimization on the molecule using the provided coordinates
        try:
            rdkitmol_optimize(mol)
            energy = rdkit_singlepoint(mol)
            status_update = 'succ'
        except RuntimeError:
            status_update = 'optfail'
            energy = None
        
        xyz, sym = rdkit_as_xyz(mol)
        optimized_p.include_from_xyz(
            xyz,
            ConformerInfo.updated_status(
                old_descr=m.descr,
                update=status_update,
                energy=energy,
            )
        )
        optimized_p.atom_symbols = initial_p.atom_symbols

    topo_checker = TopologyChecker(graph)
    for m in optimized_p:
        optimized_p.generate_connectivity(m.idx, mult=1.3)
        same_topology, extra_bonds, missing_bonds = topo_checker.same_topology(optimized_p.get_connectivity())
        if not same_topology:
            m.descr = ConformerInfo.updated_status(old_descr=m.descr, update='topofail')
            
    optimized_p.save_xyz(res_xyz)

def float_to_upper_int(x):
    if x != int(x):
        return int(x) + 1
    else:
        return int(x)

def confpool_to_batches(p, n_batches: int) -> List:
    from ringo import Confpool

    indices = [i for i in range(len(p))]
    index_batches = [
        list(index_batch)
        for index_batch in itertools.batched(indices, float_to_upper_int(len(indices) / n_batches))
    ]

    batch_p_list = []
    for index_batch in index_batches:
        batch_p = Confpool()
        for index in index_batch:
            batch_p.include_from_xyz(p[index].xyz, p[index].descr)
        batch_p.atom_symbols = p.atom_symbols
        batch_p_list.append(batch_p)
    return batch_p_list

def optimize_mmff_and_geomcheck_worker(input: Tuple[str, str, str]) -> None:
    start_xyz, initial_sdf, res_xyz = input
    optimize_mmff_and_geomcheck(
        start_xyz=start_xyz,
        initial_sdf=initial_sdf,
        res_xyz=res_xyz,
    )

def multiproc_mmff_opt(args_list: List[Tuple[str, str, str]], num_proc: int) -> None:
    with multiprocessing.Pool(processes=num_proc) as pool:
        pool.map(optimize_mmff_and_geomcheck_worker, args_list)

def optimize_gfnff_and_geomcheck(
    initial_sdf: str,
    res_xyz: str,
    xtb_dir: str,
    start_xyz: str=None,
    initial_p: str=None,
):
    XTB_TEMP_DIR = xtb_dir
    assert (initial_p is not None) ^ (start_xyz is not None)

    from ringo import Confpool
    from chemscripts.utils import write_xyz

    ccmol = ccgeom.Molecule(sdf=initial_sdf)
    graph = ccmol.G
    charge = ccmol.total_charge()

    if start_xyz is not None:
        initial_p = Confpool()
        initial_p.include_from_file(start_xyz)
    
    if start_xyz is not None:
        tempdir_firstname = ntpath.basename(start_xyz).replace('.xyz', '')
    else:
        tempdir_firstname = f'calc{random.randint(1000000,9999999)}'

    optimized_p = Confpool()

    def prep_given_index(initial_index: int) -> str:
        dirname = f"{tempdir_firstname}_{initial_index}"
        fulldir = os.path.join(XTB_TEMP_DIR, dirname)
        os.mkdir(fulldir)
        
        newxyz_name = os.path.join(fulldir, 'start.xyz')
        write_xyz(initial_p[initial_index].xyz, initial_p.atom_symbols, newxyz_name)
        return fulldir
    
    def get_xyz_and_status(calcdir: str) -> Tuple[str, str]:
        xtbopt_name = os.path.join(calcdir, 'xtbopt.xyz')
        opt_okay = os.path.exists(xtbopt_name)
        if opt_okay:    
            log_name = os.path.join(calcdir, 'log')
            opt_okay = get_xtbopt_status(log_name)
        
        if opt_okay:
            status_update = 'succ'
        else:
            status_update = 'optfail'

        if os.path.exists(xtbopt_name):
            output_xyz = xtbopt_name
        else:
            xtblast_name = os.path.join(calcdir, 'xtblast.xyz')
            if os.path.exists(xtblast_name):
                output_xyz = xtblast_name
            else:
                output_xyz = None
        return output_xyz, status_update

    def load_finished_xtbopt(calcdir: str, initial_index: int) -> None:
        output_xyz, status_update = get_xyz_and_status(calcdir)
        
        if output_xyz is not None:
            optimized_p.include_from_file(output_xyz)
        
        if output_xyz is None:
            return None

        xtb_descr = optimized_p[len(optimized_p) - 1].descr
        optconf_energy = get_energy_from_xtbdescr(xtb_descr)

        old_descr = initial_p[initial_index].descr
        optimized_p[len(optimized_p) - 1].descr = ConformerInfo.updated_status(
            old_descr=old_descr,
            update=status_update,
            energy=optconf_energy * H2KC
        )
        optimized_p.atom_symbols = initial_p.atom_symbols
    
    for cur_index in range(len(initial_p)):
        calcdir = prep_given_index(cur_index)
        subprocess.Popen(
            f"exec_xtbopt.sh {calcdir} {charge}",
            shell = True
        ).wait()
        load_finished_xtbopt(
            calcdir=calcdir,
            initial_index=cur_index
        )
        shutil.rmtree(calcdir)

    topo_checker = TopologyChecker(graph)
    for m in optimized_p:
        optimized_p.generate_connectivity(m.idx, mult=1.3)
        same_topology, extra_bonds, missing_bonds = topo_checker.same_topology(optimized_p.get_connectivity())
        if not same_topology:
            m.descr = ConformerInfo.updated_status(old_descr=m.descr, update='topofail')

    optimized_p.save_xyz(res_xyz)

def optimize_gfnff_and_geomcheck_worker(input: Tuple[str, str, str, str]) -> None:
    start_xyz, initial_sdf, res_xyz, xtb_dir = input
    optimize_gfnff_and_geomcheck(
        start_xyz=start_xyz,
        initial_sdf=initial_sdf,
        res_xyz=res_xyz,
        xtb_dir=xtb_dir,
    )

def multiproc_gfnff_opt(args_list: List[Tuple[str, str, str]], num_proc: int, xtb_dir: str) -> None:
    with multiprocessing.Pool(processes=num_proc) as pool:
        pool.map(
            optimize_gfnff_and_geomcheck_worker,
            [(*item, xtb_dir) for item in args_list]
        )

def split_testcase(testcase: str) -> Dict[str, str]:
    assert testcase[:3] in ('csd', 'pdb')
    return {
        'database': testcase[:3].upper(),
        'molname': testcase[3:],
    }

def format_testcase(testcase: str) -> str:
    split = split_testcase(testcase)
    return "'{molname}' (from {database})".format(**split)


def get_crmsd_matrix(p, mirror_match: bool, print_status: bool) -> np.ndarray:
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
        matr = p.get_rmsd_matrix(mirror_match=mirror_match, print_status=print_status)
        rmsd_matrices.append(matr)

    first_shape = np.shape(rmsd_matrices[0])
    # Ensure all shapes are the same
    for matrix in rmsd_matrices[1:]:
        assert np.shape(matrix) == first_shape, "Matrices do not have the same shape."

    # Apply element-wise maximum across the matrices
    max_matrix = np.maximum.reduce(rmsd_matrices)

    return max_matrix

def remove_unperspective_conformers(
    p,
    crmsd_matrix: np.ndarray,
    lowenergy_threshold: float,
    crmsd_cutoff: float
) -> np.ndarray:
    p['old_index'] = lambda m: float(m.idx)

    lowenergy_idxs = [
        m_index
        for m_index, m in enumerate(p)
        if ConformerInfo(description=m.descr).relenergy < lowenergy_threshold
    ]

    def not_unperspective(m) -> bool:
        cur_index = m.idx
        if cur_index in lowenergy_idxs:
            return True
        
        result = False
        for check_idx in lowenergy_idxs:
            if crmsd_matrix[cur_index, check_idx] < crmsd_cutoff:
                result = True
                break
        return result
    
    p.filter(not_unperspective)
    if len(p) == 0:
        return None
    keep_indices = [int(i) for i in p['old_index']]    
    resulting_matrix = crmsd_matrix[keep_indices][:, keep_indices]

    # ref_matrix = get_crmsd_matrix(p=p, mirror_match=True, print_status=False)
    # assert resulting_matrix.shape == ref_matrix.shape, f"Actual={resulting_matrix.shape}, Ref={ref_matrix.shape}"
    # diff_matrix = resulting_matrix - ref_matrix
    # assert diff_matrix.max() < 1e-5, f"MaxDiff = {diff_matrix.max()}"
    return resulting_matrix
