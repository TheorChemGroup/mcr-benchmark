import networkx as nx
from networkx.algorithms import isomorphism
import numpy as np

import json
import inspect
from typing import List, Tuple, Optional, Iterable, Dict, Callable

import ringo


class Molecule:
    def __init__(self, sdf: str):
        self.readsdf(sdf)

    def readsdf(self, file: str) -> None:
        with open(file, "r") as f:
            lines = f.readlines()
        natoms = int(lines[3][0:3])
        nbonds = int(lines[3][3:6])
        self.G = nx.Graph()
        for i in range(4, 4 + natoms):
            self.G.add_node(i-4)
            parts = lines[i].replace("\n", "").split()
            self.G.nodes[i-4]['xyz'] = np.array([float(parts[0]), float(parts[1]), float(parts[2])])
            self.G.nodes[i-4]['symbol'] = parts[3]
        for i in range(4 + natoms, 4 + natoms + nbonds):
            at1 = int(lines[i][0:3])
            at2 = int(lines[i][3:7])
            bondtype = int(lines[i][7:10])
            self.G.add_edge(at1 - 1, at2 - 1)
            self.G[at1 - 1][at2 - 1]['type'] = bondtype
        for line in lines:
            if 'M  CHG' in line:
                parts = line.split()[3:]
                charges = [(int(parts[i]), int(parts[i + 1])) for i in range(0, len(parts), 2)]
                for index, charge in charges:
                    self.G.nodes[index-1]['chrg'] = charge

    def ring_atoms(self):
        temp_graph = nx.Graph()
        temp_graph.add_nodes_from(self.G.nodes)
        temp_graph.add_edges_from(self.G.edges)
        temp_graph.remove_edges_from(nx.bridges(self.G))
        temp_graph.remove_nodes_from([
            node
            for node in temp_graph.nodes
            if len(list(temp_graph.neighbors(node))) == 0
        ])
        return list(temp_graph.nodes)

    def total_charge(self) -> int:
        res = 0
        for node, keys in self.G.nodes(data=True):
            if 'chrg' in keys:
                res += keys['chrg']
        return res

    def __add__(self, other):
        res = Molecule(shutup=True)
        res.G = nx.Graph()
        maxnode = None
        for node in self.G.nodes:
            res.G.add_node(node)
            res.G.nodes[node]['xyz'] = self.G.nodes[node]['xyz']
            res.G.nodes[node]['symbol'] = self.G.nodes[node]['symbol']
            if maxnode is None or maxnode < node:
                maxnode = node

        for edge in self.G.edges:
            res.G.add_edge(edge[0], edge[1])
            res.G[edge[0]][edge[1]]['type'] = self.G[edge[0]][edge[1]]['type']
        
        res.idx_map = {}
        for node in other.G.nodes:
            res.idx_map[node] = node + maxnode + 1
        print("Other G has {} nodes".format(repr(list(other.G.nodes))))
        for node in other.G.nodes:
            nodeidx = res.idx_map[node]
            res.G.add_node(nodeidx)
            res.G.nodes[nodeidx]['xyz'] = other.G.nodes[node]['xyz']
            res.G.nodes[nodeidx]['symbol'] = other.G.nodes[node]['symbol']

        for edge in other.G.edges:
            res.G.add_edge(res.idx_map[edge[0]], res.idx_map[edge[1]])
            res.G[res.idx_map[edge[0]]][res.idx_map[edge[1]]]['type'] = other.G[edge[0]][edge[1]]['type']
        return res

    def as_xyz(self):
        xyzs = []
        syms = []
        for atom in range(self.G.number_of_nodes()):
            syms.append(self.G.nodes[atom]['symbol'])
            xyzs.append(self.G.nodes[atom]['xyz'])
        return xyzs, syms

    def remove_bond(self, a, b):
        self.G.remove_edge(a - 1, b - 1)

    def has_bond(self, a, b):
        return self.G.has_edge(a - 1, b - 1)

    def xyz(self, i):
        return self.G.nodes[i - 1]['xyz']

    def save_sdf(self, sdfname):
        # Charge example "M  CHG  1  21  -1"
        lines = ["", "", ""]
        lines.append("%3d%3d  0  0  0  0  0  0  0  0999 V2000" % (self.G.number_of_nodes(), self.G.number_of_edges()))
        #   -5.5250    1.6470    1.0014 C   0  0  0  0  0  0  0  0  0  0  0  0
        for atom in range(self.G.number_of_nodes()):
            lines.append("%10.4f%10.4f%10.4f%3s  0  0  0  0  0  0  0  0  0  0  0  0" % (
                self.G.nodes[atom]['xyz'][0],
                self.G.nodes[atom]['xyz'][1],
                self.G.nodes[atom]['xyz'][2],
                self.G.nodes[atom]['symbol']))

        for edge in self.G.edges:
            lines.append("%3s%3s%3s  0" % (edge[0] + 1,
                                           edge[1] + 1,
                                           self.G[edge[0]][edge[1]]['type']))
        
        write_charges = False
        for node in self.G.nodes:
            if 'chrg' in self.G.nodes[node]:
                write_charges = True
                break
        if write_charges:
            charged_atoms = []
            for node in self.G.nodes:
                if 'chrg' not in self.G.nodes[node] or self.G.nodes[node]['chrg'] == 0:
                    continue
                charged_atoms.append((node+1, self.G.nodes[node]['chrg']))
            charge_strings = [f"{index}  {charge}" for index, charge in charged_atoms]
            charge_line = f"M  CHG  {len(charge_strings)}  {'  '.join(charge_strings)}"
            lines.append(charge_line)

        lines.append("M  END\n")

        with open(sdfname, "w") as f:
            f.write("\n".join(lines))
    
    def get_bonds(self):
        return list(self.G.edges)

    def add_dummy(self, position):
        idx = self.G.number_of_nodes()
        self.G.add_node(idx)
        self.G.nodes[idx]['xyz'] = position
        self.G.nodes[idx]['dummy'] = True
        # self.G.nodes[idx]['symbol'] = 'He'
        return idx + 1

    def remove_dummies(self):
        for i in range(self.G.number_of_nodes()):
            if 'dummy' in self.G.nodes[i] and self.G.nodes[i]['dummy']:
                self.G.remove_node(i)

    def atom_xyz(self, idx):
        return self.G.nodes[idx - 1]['xyz']
    
    def keep_atoms(self, idxs):
        for i in range(len(idxs)):
            idxs[i] -= 1
        for node in list(self.G.nodes):
            if node not in idxs:
                self.G.remove_node(node)
        i = 0
        relabel_mapping = {}
        relabel_short = {}
        node_names = sorted(list(self.G.nodes))
        for i, node in enumerate(node_names):
            relabel_mapping[node] = i
            if node in idxs:
                relabel_short[node + 1] = i + 1
        self.G = nx.relabel_nodes(self.G, relabel_mapping, copy=False)
        return relabel_short


def same_element(n1_attrib, n2_attrib):
    return 'symbol' not in n1_attrib or 'symbol' not in n2_attrib or (n1_attrib['symbol'] == n2_attrib['symbol'])

def same_bondtype(e1_attrib, e2_attrib):
    return 'type' not in e1_attrib or 'type' not in e2_attrib or (e1_attrib['type'] == e2_attrib['type'])

def plus_one(bond: Tuple[int]) -> Tuple[int]:
    return tuple(i+1 for i in bond)


def fits_the_interval(value, lower, upper):
    step = 360
    while value < lower and value < upper:
        value += step
    while value > lower and value > upper:
        value -= step
    return (value > lower) and (value < upper)

class SamplingCustomizer:
    def __init__(self, mol_sdf: str, sampling_rules: dict) -> None:
        self.mol_sdf = mol_sdf
        self.sampling_rules = sampling_rules

    def primary_locate_straight(self,
            bond: Tuple[int, int] | Tuple[int, int, int, int],
            dihedral_type: Optional[str] = None
        ) -> Tuple[str, int] | None:
        if dihedral_type is None:
            lists_to_check = self.primary_convention
        else:
            lists_to_check = {dihedral_type: self.primary_convention[dihedral_type]}

        if len(bond) == 2:
            match_condition = lambda atoms: bond[0] == atoms[1] and bond[1] == atoms[2]
        else: # if len(bond) == 4:
            match_condition = lambda atoms: bond == atoms

        match_found = False
        list_name = None
        item_index = None
        for list_name, dihedral_list in lists_to_check.items():
            for item_index, item in enumerate(dihedral_list):
                if match_condition(item['atoms']):
                    match_found = True
                    break
            if match_found:
                break
        if match_found:
            return (list_name, item_index)
        else:
            return None

    def primary_locate(self,
            bond: Tuple[int, int] | Tuple[int, int, int, int],
            dihedral_type: Optional[str] = None
        ) -> Tuple[str, int] | None:
        output = self.primary_locate_straight(bond=bond, dihedral_type=dihedral_type)
        if output is not None:
            return output
        output = self.primary_locate_straight(bond=tuple(reversed(bond)), dihedral_type=dihedral_type)
        return output

    def primary_contains(self,
            bond: Tuple[int, int] | Tuple[int, int, int, int],
            dihedral_type: Optional[str] = None
        ) -> bool:
        return self.primary_locate(bond=bond, dihedral_type=dihedral_type) is not None

    def get_fixed_bonds(self) -> List[Tuple[int, int]]:
        self.primary_convention = {
            'amide': [],
            'other': [],
        }

        trivial_amide: bool = (len(self.sampling_rules['amide_treat']['allowed_configurations']) == 0)
        trivial_other: bool = (len(self.sampling_rules['special_dihedrals']) == 0)

        if not trivial_amide or not trivial_other:
            amide_default_values: List[float] = self.sampling_rules['amide_treat']['allowed_configurations']
            amide_default_width: float = self.sampling_rules['amide_treat']['sampling_width']
            amide_default_mandatory: float = self.sampling_rules['amide_treat']['mandatory']
            amide_default_filtering_width: float = self.sampling_rules['amide_treat']['filtering_width']
            self.primary_convention['amide'] = [
                {
                    'atoms': four_atoms,
                    'values': amide_default_values,
                    'width': amide_default_width,
                    'mandatory': amide_default_mandatory,
                    'filtering_width': amide_default_filtering_width,
                }
                for four_atoms in self.get_amide_bonds(self.mol_sdf)
            ]

        if not trivial_other:
            for item in self.sampling_rules['special_dihedrals']:
                bond = item['bond']

                if len(bond) == 2:
                    assert self.primary_contains(bond, dihedral_type='amide'), \
                        f"Bond '{plus_one(bond)}' does not correspond to amide dihedral C(=O)-N(H)R, thus, it has to be passed by 4 atom indices."
                else:
                    assert not self.primary_contains(bond, dihedral_type='other'), \
                        f"Multiple declarations of requirements for sampling of dihedral '{plus_one(bond)}'"
                
                amide_search_result = self.primary_locate(bond, dihedral_type='amide')
                if amide_search_result is not None:
                    _, item_index = amide_search_result
                    amide_entry = self.primary_convention['amide'][item_index]
                    if len(bond) == 4: # If the user likes non-standard E/Z nomenclature, let it be
                        amide_entry['atoms'] = item['atoms']
                    amide_entry['values'] = item['allowed_configurations']
                    amide_entry['width'] = item['sampling_width']
                    amide_entry['mandatory'] = item['mandatory']
                    amide_entry['filtering_width'] = item['filtering_width']

                    # If the 'amide_entry' is going to be deleted, load it into 'other'
                    if trivial_amide:
                        self.primary_convention['other'].append(amide_entry)
                else:
                    self.primary_convention['other'].append({
                        'atoms': item['bond'],
                        'values': item['allowed_configurations'],
                        'width': item['sampling_width'],
                        'mandatory': item['mandatory'],
                        'filtering_width': item['filtering_width'],
                    })
        
        # If amide dihedrals do not have to be constrained, remove all their data
        if trivial_amide:
            self.primary_convention['amide'] = []
        
        bonds_list = [
            (item['atoms'][1] + 1, item['atoms'][2] + 1)
            for dihedral_list in self.primary_convention.values()
            for item in dihedral_list
        ]
        verification_set = set(
            tuple(sorted(list(pair)))
            for pair in bonds_list
        )
        assert len(verification_set) == len(bonds_list), \
            f"Duplicates were found in the prepared list of constrained dihedrals (add 1 to each atom index): {bonds_list}"
        return bonds_list

    def create_confpool_from_sdf(self) -> ringo.Confpool:
        mol = Molecule(sdf=self.mol_sdf)
        p = ringo.Confpool()
        xyz, sym = mol.as_xyz()
        p.include_from_xyz(xyz, '')
        p.atom_symbols = sym
        return p

    def set_sampling_limits(self, mol: ringo.Molecule) -> None:
        DEG2RAD: float = ringo.DEG2RAD
        RAD2DEG: float = ringo.RAD2DEG

        dofs_list, _ = mol.get_ps()
        custom_dof_limits = {}

        self.filter_dihedrals = []
        self.skip_dihedrals = []

        # # CHECK: Turn off verification for release
        # test_p: ringo.Confpool = self.create_confpool_from_sdf()
        # for list_name, dihedral_list in self.primary_convention.items():
        #     for request_index, dihedral_request in enumerate(dihedral_list):
        #         req_atoms = dihedral_request['atoms']
        #         for dof_index, dof_atoms in enumerate(dofs_list):
        #             if (req_atoms[1], req_atoms[2]) != (dof_atoms[1], dof_atoms[2]) and \
        #                 (req_atoms[1], req_atoms[2]) != (dof_atoms[2], dof_atoms[1]):
        #                 continue
                    
        #             req_dihedral = test_p[0].z(*plus_one(req_atoms))
                    
        #             correct_value = test_p[0].z(*plus_one(dof_atoms)) * RAD2DEG
        #             value_to_check = mol.translate_dihedral(req_atoms, req_dihedral, dof_atoms) * RAD2DEG

        #             assert abs(correct_value - value_to_check) < 0.01, \
        #                 f"correct_value={correct_value}, value_to_check={value_to_check}"

        # Going to replace Ringo-generated warnings with better ones
        unmet_bond_requests = []
        new_warnings = []
        for list_name, dihedral_list in self.primary_convention.items():
            for request_index, dihedral_request in enumerate(dihedral_list):
                req_atoms = dihedral_request['atoms']
                req_values = dihedral_request['values']
                req_width = dihedral_request['width']
                req_mandatory = dihedral_request['mandatory']
                request_satisfied = False
                for dof_index, dof_atoms in enumerate(dofs_list):
                    if (req_atoms[1], req_atoms[2]) != (dof_atoms[1], dof_atoms[2]) and \
                        (req_atoms[1], req_atoms[2]) != (dof_atoms[2], dof_atoms[1]):
                        continue
                    
                    custom_dof_limits[dof_index] = [
                        [
                            mol.translate_dihedral(req_atoms, req_boundary, dof_atoms, normalize_result=False)
                            for req_boundary in [(req_value - req_width / 2) * DEG2RAD, (req_value + req_width / 2) * DEG2RAD]
                        ]
                        for req_value in req_values
                    ]
                    request_satisfied = True
                
                if not request_satisfied:
                    dof_type: str = mol.get_dof_type(req_atoms[1], req_atoms[2])
                    
                    # Needed to get indices of Ringo-generated warnings
                    unmet_bond_requests.append((req_atoms[1] + 1, req_atoms[2] + 1))

                    if dof_type == 'dep':
                        if not req_mandatory:
                            message_data = {
                                "message": f"Requested dihedral {plus_one(req_atoms)} is flexible but cannot be made a degree of freedom. Use 'mandatory' option to enforce this dihedral via conformer filtering.",
                                "subject": ringo.UNMET_DOF_REQUEST + "[important]",
                                "atoms": [req_atoms[1], req_atoms[2]],
                                "file": __file__,
                                "line": inspect.currentframe().f_lineno,
                            }
                            new_warnings.append(message_data)
                            self.skip_dihedrals.append((list_name, request_index))
                        else:
                            self.filter_dihedrals.append((list_name, request_index))
                    else:
                        assert dof_type == 'fixed', f"Unexpected DOF type corresponding to bond {req_atoms[1] + 1}-{req_atoms[2] + 1}"

                        if req_mandatory:
                            message_data = {
                                "message": f"Cannot modify configuration of the requested mandatory dihedral {plus_one(req_atoms)} since it belongs to kinematically rigid fragment (either rigid cycle or non-single bond).",
                                "subject": ringo.UNMET_DOF_REQUEST + "[important]",
                                "atoms": [req_atoms[1], req_atoms[2]],
                                "file": __file__,
                                "line": inspect.currentframe().f_lineno,
                            }
                            new_warnings.append(message_data)
                        self.skip_dihedrals.append((list_name, request_index))

        # print(repr(self.primary_convention))
        # print(repr(custom_dof_limits))
        mol.customize_sampling(custom_dof_limits)

        self.warnings_to_skip = []
        warnings = ringo.get_status_feed(important_only=True)
        for i, item in enumerate(warnings):
            atoms = tuple(item['atoms'])
            if item['subject'] == 'Unfulfilled DOF request' and \
                (atoms in unmet_bond_requests or (atoms[1], atoms[0]) in unmet_bond_requests):
                self.warnings_to_skip.append(i)
        
        for message_json in new_warnings:
            ringo.add_message_to_feed(json.dumps(message_json))

    def get_filtering_function(self) -> Callable[[ringo.MolProxy], bool]:
        if len(self.filter_dihedrals) > 0:
            DEG2RAD: float = ringo.DEG2RAD

            manual_dihedrals = {}
            for list_name, request_index in self.filter_dihedrals:
                dihedral_request = self.primary_convention[list_name][request_index]
                req_atoms = dihedral_request['atoms']
                req_values = dihedral_request['values']
                req_filtering_width = dihedral_request['filtering_width']
                manual_dihedrals[req_atoms] = tuple(
                    ((req_value - req_filtering_width / 2) * DEG2RAD, (req_value + req_filtering_width / 2) * DEG2RAD)
                    for req_value in req_values
                )
            return {'filter': manual_dihedrals}
        else:
            return {}

    # def validate_ensemble(self, p: ringo.Confpool) -> None: # CHECK: Hide validation
    #     RAD2DEG: float = ringo.RAD2DEG
    #     test_passed = True
    #     error_messages = []
    #     for m in p:
    #         for list_name, dihedral_list in self.primary_convention.items():
    #             for request_index, dihedral_request in enumerate(dihedral_list):
    #                 if (list_name, request_index) in self.skip_dihedrals:
    #                     continue
                    
    #                 req_atoms = dihedral_request['atoms']
    #                 req_values = dihedral_request['values']
    #                 req_width = dihedral_request['width']
    #                 actual_value = m.z(*plus_one(req_atoms)) * RAD2DEG
                    
    #                 if all(
    #                     not fits_the_interval(actual_value, lower=(req_value - req_width / 2), upper=(req_value + req_width / 2))
    #                     for req_value in req_values
    #                 ):
    #                     error_messages.append(f"Failed to impose constraint on dihedral {plus_one(req_atoms)}. Value = {actual_value}\n")
    #                     test_passed = False
    #         if not test_passed:
    #             break
    #     assert test_passed, f"Failed to impose one or several dihedral constraints:\n{''.join((('%s. ' % (i+1)) + message) for i, message in enumerate(error_messages))}"
                    

    @staticmethod
    def get_amide_bonds(
            mol_sdf: str
        ) -> List[Tuple[int, int, int, int]]:
        start_mol = Molecule(sdf=mol_sdf)
        mol_graph: nx.Graph = start_mol.G

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
        
        generic_amide_group = nx.Graph()
        generic_amide_group.add_node(0, symbol='C')
        generic_amide_group.add_node(1, symbol='N')
        generic_amide_group.add_node(2, symbol='O')
        generic_amide_group.add_node(3)
        generic_amide_group.add_node(4)
        generic_amide_group.add_edge(0, 1, type=1)
        generic_amide_group.add_edge(0, 2, type=2)
        generic_amide_group.add_edge(3, 1, type=1)
        generic_amide_group.add_edge(4, 1, type=1)

        def load_amide_dihedrals(amide_subgraph: nx.Graph, dihedral_container, dihedral_indices, valence_requests: Dict[int, int]={}) -> None:
            # Initialize the subgraph isomorphism matcher
            matcher = isomorphism.GraphMatcher(mol_graph, amide_subgraph, node_match=same_element, edge_match=same_bondtype)
            
            # Find all matches of the subgraph in the larger graph
            for match in matcher.subgraph_isomorphisms_iter():
                rev_match = {value: key for key, value in match.items()}
                new_dihedral = tuple(
                    rev_match[subgraph_index]
                    for subgraph_index in dihedral_indices
                )

                dihedral_already_present: bool = any(
                    (new_dihedral[1], new_dihedral[2]) == (older_dihedral[1], older_dihedral[2])
                    for older_dihedral in dihedral_container
                )
                if dihedral_already_present:
                    continue

                valence_accepted: bool = all(
                    len(list(mol_graph.neighbors(rev_match[subgraph_index]))) == required_valence
                    for subgraph_index, required_valence in valence_requests.items()
                )
                if not valence_accepted:
                    continue

                dihedral_container.append(new_dihedral)


        full_amide_dihedrals = []
        load_amide_dihedrals(
            amidegroup,
            full_amide_dihedrals,
            dihedral_indices=[
                2, # oxygen
                0, # carbon
                1, # nitrogen
                4, # carbon2
            ],
            valence_requests={0: 3, 1: 3} # Consider trivalent nitrogen and carbonyl carbon only
        )
        load_amide_dihedrals(
            generic_amide_group,
            full_amide_dihedrals,
            dihedral_indices=[
                2, # oxygen
                0, # carbon
                1, # nitrogen
                4, # any other atom
            ],
            valence_requests={0: 3, 1: 3} # Consider trivalent nitrogen and carbonyl carbon only
        )
        
        return full_amide_dihedrals
