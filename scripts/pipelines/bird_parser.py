#
# HERE WE PREPARE 'BIRD' TESTSET STRUCTURES
#

import os
import sys
import subprocess
import shlex
import numpy as np
import networkx as nx
import sklearn.decomposition
from networkx.algorithms import isomorphism
from typing import Tuple, List, Dict, Literal, Callable

from pysquared import Transformator, Transform
import pysquared.transforms.transform_templates as templates

from chemscripts.utils import H2KC
from chemscripts import utils as ccutils
from chemscripts import geom as ccgeom
from chemscripts.geom import Molecule
import networkx as nx

from utils import confsearch
from utils.confsearch import GJF_TEMPL, GAUSSIAN_PARAMETERS, fix_element_str, assert_path, include_if_exists


def birdparser_dataitems() -> dict[str, dict]:
    return {
        # # Prepare Bird
        # 'testset_archive': {'type': 'file', 'mask': './testsets/{testset}/{archivename}.tar.gz'},
        # 'testset_raw_extracted': {'type': 'file', 'mask': './testsets/{testset}/extracted_files/{archivename}_{testcase}.{ext}'},
        # 'all_testmols_sdfs': {'type': 'file', 'mask': './testsets/{testset}/start_testmols/{testcase}.sdf'},
        # 'testmols_sdfs': {'type': 'file', 'mask': './testsets/{testset}/selected_testmols/{testcase}.sdf'},
        # # Summarize
        # 'full_summary_object': {'type': 'object', 'keys': ['testset', 'testcase']},
        # 'full_summary_block': {'type': 'object', 'keys': ['testset']},
        # 'full_testset_summary': {'type': 'file', 'mask': './testsets/{testset}/full_testset_summary.xlsx'},
        # 'summary_object': {'type': 'object', 'keys': ['testset', 'testcase']},
        # 'summary_block': {'type': 'object', 'keys': ['testset']},
        # 'testset_summary': {'type': 'file', 'mask': './testsets/{testset}/testset_summary.xlsx'},
        # # Rdkit & Gaussian verification
        # 'testcase_str': {'type': 'object', 'keys': ['testset', 'testcase']},
        # 'rdkit_status': {'type': 'object', 'keys': ['testset', 'testcase']},
        # 'gauss_status': {'type': 'object', 'keys': ['testset', 'testcase']},
        # 'aug_summary_object': {'type': 'object', 'keys': ['testset', 'testcase']},
        # 'aug_summary_block': {'type': 'object', 'keys': ['testset']},
        # 'num_of_automorphisms': {'type': 'object', 'keys': ['testset', 'testcase']},
        'aug_testset_summary': {'type': 'file', 'mask': './testsets/{testset}/testset_summary_verified.xlsx'},
        # 'gaussian_sp_logs': {'type': 'file', 'mask': './testsets/{testset}/gaussian_check/{testcase}.log'},
        # 'final_testmols_sdfs': {'type': 'file', 'mask': './testsets/{testset}/corrected_testmols/{testcase}.sdf'},

        # # In cases when optimization of 'final_testmols_sdfs' has failed
        # 'fail_optimized_testmols_sdfs': {'type': 'file', 'mask': './testsets/{testset}/testmols_target/{testcase}_optfail.sdf'},

        # Final structures of the testset
        'optimized_testmols_sdfs': {'type': 'file', 'mask': './testsets/{testset}/testmols_target/{testcase}.sdf'},
        'randomized_testmols_sdfs': {'type': 'file', 'mask': './testsets/{testset}/testmols_start/{testcase}.sdf'},
    }


def get_basic_testset_summary(sdf_name: str, testcase: str) -> dict:
    import ringo
    ringo.cleanup()
    ringo.clear_status_feed()
    
    mol = ringo.Molecule(sdf=sdf_name)
    warnings = ringo.get_status_feed(important_only=True)
    return {
        'testcase': testcase,
        'warnings': repr(warnings),
        **{
            key: value
            for key, value in ringo.get_molecule_statistics(mol).items()
            if key != 'composition'
        }
    }


def generic_summary_generation_blueprint(
        grouped_transforms: Dict[Literal['get', 'merge', 'save'], str],
        item_names: Dict[Literal['inputs', 'summary', 'block', 'xlsx'], str | List[str]],
        summary_generator: Callable[..., Dict],
        testcase_key: str='testcase',
        main_block_name: str='Main'
    ) -> Tuple[Transform]:
    return (
        templates.map(grouped_transforms['get'],
            input=item_names['inputs'], output=item_names['summary'], aware_keys=[testcase_key],
            mapping=summary_generator#, forward_args=['thread_manager']
        ),
        templates.select(grouped_transforms['merge'],
            input=item_names['summary'], output=item_names['block'],
            select_method=lambda data: [obj for obj, keys in data],
            merged_keys=[testcase_key]
        ),
        templates.construct_excel_sheet(grouped_transforms['save'],
            input=item_names['block'], xlsx=item_names['xlsx'],
            block_name=lambda *args: main_block_name
        ),
    )

def primary_summary_generation_blueprint(
        item_names: Dict[Literal['sdf', 'summary', 'block', 'xlsx'], str],
        grouped_transforms: Dict[Literal['get', 'merge', 'save'], str],
        testcase_key: str='testcase',
        main_block_name: str='Main'
    ) -> Tuple[Transform]:
    other_item_names = {
        key: value
        for key, value in item_names.items()
        if key != 'sdf'
    }
    return generic_summary_generation_blueprint(
        item_names={
            'inputs': [item_names['sdf']],
            **other_item_names
        },
        summary_generator=lambda **kwargs: get_basic_testset_summary(
            sdf_name=kwargs[item_names['sdf']],
            testcase=kwargs[testcase_key]
        ),
        grouped_transforms=grouped_transforms,
        testcase_key=testcase_key,
        main_block_name=main_block_name,
    )


def final_summary_generation_blueprint(
        item_names: Dict[Literal['sdf', 'num_isom', 'old_summary', 'new_summary', 'other_items', 'block', 'xlsx'], str | List[str]],
        grouped_transforms: Dict[Literal['calc_isom', 'get', 'merge', 'save'], str],
        testcase_key: str='testcase',
        main_block_name: str='Main'
    ) -> Tuple[Transform]:
    other_item_names = {
        'summary': item_names['new_summary'],
        'block': item_names['block'],
        'xlsx': item_names['xlsx'],
    }
    return (
        templates.map(grouped_transforms['calc_isom'],
            input=item_names['sdf'], output=item_names['num_isom'],
            mapping=lambda **kw: confsearch.sdf_to_confpool(kw[item_names['sdf']]).get_num_isomorphisms()
        ),
        *generic_summary_generation_blueprint(
            item_names={
                'inputs': [
                    item_names['old_summary'],
                    item_names['num_isom'],
                    *item_names['other_items']
                ],
                **other_item_names
            },
            summary_generator=lambda **kwargs: {
                **{
                    key: kwargs[key]
                    for key in item_names['other_items']
                },
                'num_automorphisms': kwargs[item_names['num_isom']],
                **kwargs[item_names['old_summary']]
            },
            grouped_transforms={
                key: value
                for key, value in grouped_transforms.items()
                if key != 'calc_isom'
            },
            testcase_key=testcase_key,
            main_block_name=main_block_name,
        )
    )


def birdparser_transformator(ds, main_logger) -> Transformator:
    def birdcif_to_sdf(testset_raw_extracted, all_testmols_sdfs):
        for cifname, keys in testset_raw_extracted:
            with open(cifname, 'r') as f:
                ciflines = f.readlines()

            startline_idx = None
            endline_idx = None
            for i, line in enumerate(ciflines):
                if '_chem_comp_atom' in line and '_chem_comp_atom' not in ciflines[i + 1]:
                    startline_idx = i + 1
                elif startline_idx is not None and line.startswith('#'):
                    endline_idx = i
                    break
            assert startline_idx is not None and endline_idx is not None

            mol = ccgeom.Molecule(shutup=True)
            mol.G = nx.Graph()
            abort = False
            cif_signs = {}
            for line in ciflines[startline_idx:endline_idx]:
                parts = line.split()
                if parts[9] == '?' or parts[10] == '?' or parts[11] == '?':
                    abort = True
                    break
                
                atom_idx = mol.G.number_of_nodes()
                cif_signs[parts[1]] = atom_idx
                mol.G.add_node(atom_idx,
                    symbol=fix_element_str(parts[3]),
                    xyz=[
                        float(parts[9]),
                        float(parts[10]),
                        float(parts[11]),  
                    ])
            
            if abort:
                return

            startline_idx = None
            endline_idx = None
            for i, line in enumerate(ciflines):
                if '_chem_comp_bond' in line and '_chem_comp_bond' not in ciflines[i + 1]:
                    startline_idx = i + 1
                elif startline_idx is not None and line.startswith('#'):
                    endline_idx = i
                    break
            assert startline_idx is not None and endline_idx is not None

            BONDTYPES = {
                'SING': 1,
                'DOUB': 2,
                'TRIP': 3,
            }
            for line in ciflines[startline_idx:endline_idx]:
                parts = line.split()
                mol.G.add_edge(
                    cif_signs[parts[1]], 
                    cif_signs[parts[2]],
                    type=BONDTYPES[parts[3]]
                )
            sdfname = all_testmols_sdfs.get_path()
            mol.save_sdf(sdfname=sdfname)
            all_testmols_sdfs.include_element(sdfname)

    # These test cases are too bad and cannot be easily fixed
    banned_ids = [
        # Wrongly positioned hydrogens
        'PRDCC002301', 'PRDCC001255', 'PRDCC000901', 'PRDCC001172', 'PRDCC001072', 'PRDCC001073', 'PRDCC001074', 'PRDCC000223', 'PRDCC000759', 'PRDCC000760', 'PRDCC000225', 'PRDCC000221',
        # Unable to optimize
        'PRDCC000723', 'PRDCC000724', 'PRDCC000726', 'PRDCC000727', 'PRDCC000728', 'PRDCC000762', 'PRDCC001233', 'PRDCC001234', 'PRDCC001235', 'PRDCC001236',
        # Too many isomorphisms
        'PRDCC002527',
    ]

    def check_if_cyclic(sdf_name):
        if any(
            f'{banned_case}.sdf' in sdf_name
            for banned_case in banned_ids
        ):
            return False
        
        mol = ccgeom.Molecule(sdf=sdf_name)
        graph = nx.Graph()
        graph.add_edges_from(mol.G.edges)
        bridges = list(nx.bridges(graph))
        graph.remove_edges_from(bridges)

        macrocycle_found = False
        for comp in nx.connected_components(graph):
            if len(comp) > 6:
                macrocycle_found = True
                break
        
        if macrocycle_found:
            import ringo
            ringo.cleanup()
            ringo.clear_status_feed()
            mol = ringo.Molecule(sdf=sdf_name)
            # assert len(warnings) == 0, f"Got Ringo warnings for '{sdf_name}':\n{repr(warnings)}"
            mol_stats = ringo.get_molecule_statistics(mol)
            ringo.cleanup()
            if mol_stats['num_cyclic_dofs'] > 0 and mol_stats['num_dofs'] <= 80:
                return True
        return False

    from .replacement_fragments import get_molgraph_replacements
    molgraph_corrections = get_molgraph_replacements()

    def fix_testmol_topology(input_file, res_filename):
        ccmol = ccgeom.Molecule(sdf=input_file)
        graph = ccmol.G

        for atom, keys in graph.nodes(data=True):
            if keys['symbol'] != 'N':
                continue
                
            valence = 0
            for nb in graph.neighbors(atom):
                valence += graph[atom][nb]['type']
            assert valence <= 4, f"Valence of N{atom+1} is {valence}"

            if valence == 4:
                keys['chrg'] = 1

        # if 'PRDCC002379' in input_file:
        for cycle in nx.minimum_cycle_basis(graph):
            if len(cycle) > 7 or len(cycle) == 3:
                continue
            subgraph = graph.subgraph(cycle)

            vertices = [
                keys['xyz'].copy()
                for node, keys in subgraph.nodes(data=True)
            ]
            medoid = sum(vertices)
            medoid /= len(vertices)
            for item in vertices:
                item -= medoid

            pca = sklearn.decomposition.PCA(3)
            pca.fit(vertices)
            ring_orthogonal = pca.components_[2] # Smallest singular value

            deviations = [
                atom_direction @ ring_orthogonal
                for atom_direction in vertices
            ]
            
            # Planarity criterion
            is_aromatic = all(
                abs(deviation) < 0.05
                for deviation in deviations
            )

            # 4-valent carbon => non-aromatic
            for atom_idx, keys in subgraph.nodes(data=True):
                if keys['symbol'] != 'C':
                    continue
                outside_neighbors = [
                    node
                    for node in graph.neighbors(atom_idx)
                    if node not in cycle
                ]
                assert len(outside_neighbors) in (1, 2)
                if len(outside_neighbors) == 2:
                    is_aromatic = False
                    
            if not is_aromatic:
                continue
            
            n_electrons = 0
            questionable_nitrogens = []
            # ic(input_file)
            for atom_idx, keys in subgraph.nodes(data=True):
                outside_neighbors = [
                    node
                    for node in graph.neighbors(atom_idx)
                    if node not in cycle
                ]
                if keys['symbol'] == 'C':
                    assert len(outside_neighbors) == 1, "Unexpected number of outside neighbors " \
                        f"({len(outside_neighbors)} != 1) of atom C{atom_idx + 1}"
                    # ic(f"{n_electrons} += 1")
                    n_electrons += 1
                elif keys['symbol'] in ('O', 'S', 'Se', 'Te'):
                    assert len(outside_neighbors) == 0, "Unexpected number of outside neighbors " \
                        f"({len(outside_neighbors)} != 0) of atom C{atom_idx + 1}"
                    # ic(f"{n_electrons} += 2")
                    n_electrons += 2
                elif keys['symbol'] in ('N', 'P', 'As', 'Sb'):
                    assert len(outside_neighbors) in (0, 1), "Unexpected number of outside neighbors " \
                        f"({len(outside_neighbors)} != 0, 1 or 2) of atom C{atom_idx + 1}"
                    if len(outside_neighbors) == 1:
                        # ic(f"{n_electrons} += 2")
                        n_electrons += 2
                        if graph.nodes[outside_neighbors[0]]['symbol'] == 'H':
                            questionable_nitrogens.append({
                                'N': atom_idx,
                                'H': outside_neighbors[0],
                            })
                    elif len(outside_neighbors) == 0:
                        # ic(f"{n_electrons} += 1")
                        n_electrons += 1
                if 'chrg' in keys:
                    # ic(f"{n_electrons} -= keys['chrg']")
                    n_electrons -= keys['chrg']

            remove_hydrogens = []
            for atoms in questionable_nitrogens:
                n_idx, h_idx = atoms['N'], atoms['H']
                ring_nbs = [
                    node
                    for node in subgraph.neighbors(n_idx)
                ]
                nb1_xyz = subgraph.nodes[ring_nbs[0]]['xyz']
                nb2_xyz = subgraph.nodes[ring_nbs[1]]['xyz']
                n_xyz = subgraph.nodes[n_idx]['xyz']

                a_dir = nb1_xyz - n_xyz
                b_dir = nb2_xyz - n_xyz
                normal_dir = np.cross(a_dir, b_dir)
                normal_dir /= np.linalg.norm(normal_dir)

                h_dir = graph.nodes[h_idx]['xyz'] - n_xyz
                angle = ccutils.get_angle(dirs=(normal_dir, h_dir))
                deviation = abs(angle - 90.0)
                if deviation > 15.0:
                    remove_hydrogens.append(h_idx)
                    if 'chrg' in graph.nodes[n_idx]:
                        graph.nodes[n_idx]['chrg'] -= 1
                    n_electrons -= 1

            if len(remove_hydrogens) > 0:
                graph.remove_nodes_from(remove_hydrogens)
                # ic(input_file)
                # ic(cycle)
                # ic(questionable_nitrogens)
                # ic(remove_hydrogens)
                # ic(n_electrons)
                if 'PRDCC000006.sdf' not in input_file:
                    assert n_electrons == 6
        
        def same_element(n1_attrib, n2_attrib):
            if 'symbol' in n1_attrib and 'symbol' in n2_attrib:
                return n1_attrib['symbol'] == n2_attrib['symbol']
            else:
                return True

        def same_bondtype(e1_attrib, e2_attrib):
            if 'type' in e1_attrib and 'type' in e2_attrib:
                return e1_attrib['type'] == e2_attrib['type']
            else:
                return True

        for fragment_data in molgraph_corrections:
            subgraph = fragment_data['subgraph']
            charges = fragment_data['charges']
            valence_data = fragment_data['check_valence']
            matcher = isomorphism.GraphMatcher(graph, subgraph, node_match=same_element, edge_match=same_bondtype)

            processed_atoms = []
            protected_atoms = []
            remove_atoms = []
            for match in matcher.subgraph_isomorphisms_iter():
                rev_match = {value: key for key, value in match.items()}

                proceed = True
                for sub_idx in charges.keys():
                    full_idx = rev_match[sub_idx]
                    if full_idx in processed_atoms:
                        proceed = False
                        break
                if 'protect_atoms' in fragment_data:
                    for sub_idx in fragment_data['protect_atoms']:
                        full_idx = rev_match[sub_idx]
                        if full_idx in protected_atoms:
                            proceed = False
                            break
                for sub_idx, valence in valence_data:
                    full_idx = rev_match[sub_idx]
                    if len(list(graph.neighbors(full_idx))) != valence:
                        proceed = False
                        break
                if not proceed:
                    continue

                for sub_idx, charge in charges.items():
                    full_idx = rev_match[sub_idx]
                    # assert 'chrg' not in graph.nodes[full_idx], f"Charge of atom {full_idx+1} is already set to {graph.nodes[full_idx]['chrg']}"
                    graph.nodes[full_idx]['chrg'] = charge
                    # print(f"Setting the charge of {full_idx+1} to {charge}")
                    processed_atoms.append(full_idx)
                
                if 'protect_atoms' in fragment_data:
                    for sub_idx in fragment_data['protect_atoms']:
                        full_idx = rev_match[sub_idx]
                        protected_atoms.append(full_idx)
            
                if 'fix_bondtypes' in fragment_data:
                    fixbonds_data = fragment_data['fix_bondtypes']
                    for atA, atB, newtype in fixbonds_data:
                        full_idxA = rev_match[atA]
                        full_idxB = rev_match[atB]
                        graph[full_idxA][full_idxB]['type'] = newtype
                
                if 'remove_atoms' in fragment_data:
                    for at_idx in fragment_data['remove_atoms']:
                        full_idx = rev_match[at_idx]
                        assert graph.nodes[full_idx]['symbol'] == 'H'
                        remove_atoms.append(full_idx)
            graph.remove_nodes_from(remove_atoms)

        ccmol.G = nx.convert_node_labels_to_integers(graph)
        ccmol.save_sdf(res_filename)
        return res_filename

    def verify_rdkit():
        import sys
        from ..utils.confsearch import rdkitmol_from_sdf
        rdkitmol_from_sdf(sys.argv[1])

    script_code = templates.function_to_script(verify_rdkit)
    with open('verify_rdkit.py', 'w') as f: # TODO This is bad. Shouldn't produce garbage in the script dir
        f.write(script_code)

    def verify_testcase(sdf, warning_message, testcase):
        command = f"{sys.executable} verify_rdkit.py {sdf.access_element()}"
        result = subprocess.run(
            shlex.split(command),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        error_lines = [
            line
            for line in str(result.stderr.decode('utf-8')).split('\n')
            if 'error' in line.lower() or 'exception' in line.lower()
        ]
        error_message = '  '.join(error_lines)
        if len(error_message) > 0:
            warning_message.include_element(error_message)
        else:
            warning_message.include_element('Okay')

    def get_gaussian_status(log):
        if ccutils.is_normal_termination(log, '.gjf'):
            return 'Okay'
        else:
            lines = open(log, 'r').readlines()
            for line in lines:
                if "The combination of multiplicity" in line and "electrons is impossible" in line:
                    return line
                if "Atom" in line and "has no semi-empirical parameters." in line:
                    return 'Okay (metal?)'

    def optimize_initial_testconfs():
        import sys
        module_dir = INSERT_HERE
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        from ..utils.confsearch import rdkitmol_from_sdf, rdkitmol_optimize, rdkit_save_sdf

        input_sdf, output_sdf = sys.argv[1], sys.argv[2]
        mol = rdkitmol_from_sdf(input_sdf, load_xyz=True)
        assert mol is not None
        try:
            rdkitmol_optimize(mol)
            rdkit_save_sdf(mol, output_sdf, template_sdf=input_sdf)
        except:
            rdkit_save_sdf(mol, sys.argv[3], template_sdf=input_sdf)

    def randomize_testconfs():
        import sys, os, ntpath
        module_dir = INSERT_HERE
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        os.chdir(os.path.dirname(__file__))

        import ringo
        from chemscripts import geom as ccgeom
        from ..utils.confsearch import print_ringo_warnings, get_amide_bonds, sdf_to_confpool, check_dihedral_limits, \
            rdkitmol_from_sdf, rdkit_add_coordinates, rdkitmol_optimize, rdkit_as_xyz

        from icecream import install
        install()

        start_sdf = sys.argv[1]
        num_automorphisms = int(sys.argv[2])
        result_sdf = sys.argv[3]
        molname = ntpath.basename(start_sdf)
        ic(start_sdf, num_automorphisms, result_sdf)
        
        # ringo.set_radius_multiplier(0.5)

        # Amide bonds have to be sampled in vicinity of the original dihedral
        fixed_dihedrals, full_fixed_dihedrals = get_amide_bonds(start_sdf)

        mol = ringo.Molecule(sdf=start_sdf, request_free=fixed_dihedrals) # , require_best_sequence=True
        print_ringo_warnings('initialization', molname)
        dofs_list, dofs_values = mol.get_ps()
        
        custom_dof_limits = {}
        manual_dihedrals = {}
        confcompare_p = sdf_to_confpool(start_sdf)
        warnings = ringo.get_status_feed(important_only=False)
        for req_dih, full_req_dihedral in zip(fixed_dihedrals, full_fixed_dihedrals):
            found_dof = False
            for i, item in enumerate(dofs_list):
                if (req_dih[0], req_dih[1]) == (item[1]+1, item[2]+1) or \
                    (req_dih[0], req_dih[1]) == (item[2]+1, item[1]+1):
                    found_dof = True
                    assert abs(confcompare_p[0].z(*[i+1 for i in item]) - dofs_values[i]) < 0.01
                    custom_dof_limits[i] = [dofs_values[i] - 10*ringo.DEG2RAD, dofs_values[i] + 10*ringo.DEG2RAD] # Normalize?
                    break
            if found_dof:
                continue

            found_warning = False
            for item in warnings:
                if item['subject'] != ringo.UNMET_DOF_REQUEST:
                    continue
                if req_dih[0] in item['atoms'] and req_dih[1] in item['atoms']:
                    found_warning = True
            assert found_warning, f"Cannot find UNMET_DOF_REQUEST warning for bond {req_dih[0]}-{req_dih[1]}"
            manual_dihedrals[full_req_dihedral] = confcompare_p[0].z(*full_req_dihedral)
            print(f"Unable to enforce {repr(full_req_dihedral)}.")
        if 'PRDCC000211.sdf' not in start_sdf:
            mol.customize_sampling(custom_dof_limits)
        ic(molname, custom_dof_limits)
        ic(molname, manual_dihedrals)

        MIN_RMSD_ALLOWED = 2.0

        def found_appropriate(test_p, ref_p, save_sdf) -> bool:
            rdkit_mol = rdkitmol_from_sdf(start_sdf)
            acceptable_confs = {}
            for m in test_p:
                rdkit_add_coordinates(rdkit_mol, m.xyz)
                rdkitmol_optimize(rdkit_mol)
                xyz, syms = rdkit_as_xyz(rdkit_mol)
                assert syms == test_p.atom_symbols

                cur_mol_idx = len(ref_p)
                ref_p.include_from_xyz(xyz, '')
                cur_rmsd, _, __ = ref_p[0].rmsd(ref_p[cur_mol_idx])
                if cur_rmsd > MIN_RMSD_ALLOWED:
                    acceptable_confs[cur_mol_idx] = cur_rmsd
                
            if len(acceptable_confs) == 0:
                return False
            
            largest_rmsd_index = max(acceptable_confs, key=lambda index: acceptable_confs[index])
            ccmol = ccgeom.Molecule(sdf=start_sdf)
            ccmol.from_xyz(ref_p[largest_rmsd_index].xyz, ref_p.atom_symbols)
            ccmol.save_sdf(save_sdf)
            return True

        terminate = False
        while not terminate:
            p = ringo.Confpool()
            if num_automorphisms > 500:
                start_mol = ccgeom.Molecule(sdf=start_sdf)
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
                rmsd_settings = {
                        'isomorphisms': {
                            'ignore_elements': ['HCarbon'],
                        },
                        'rmsd': {
                            'threshold': 0.2,
                            'mirror_match': True,
                        }
                    }

            mcr_kwargs = {
                'rmsd_settings': rmsd_settings,
                'nthreads': 16,
                'max_conformers': 1,
                # 'timelimit': 60,
            }
            if len(manual_dihedrals) > 0:
                def filter_function(m):
                    for dihedral, value in manual_dihedrals.items():
                        if abs(m.z(*dihedral) - value) > 10.0:
                            return False
                    return True
                if 'PRDCC000211.sdf' not in start_sdf:
                    mcr_kwargs['filter'] = filter_function
            ic(mcr_kwargs)
            print("Starting confsearch")
            res = ringo.run_confsearch(mol, pool=p, **mcr_kwargs)
            ic(res)
            print(f"Generated {len(p)} conformers of {ntpath.basename(start_sdf)}")
            print_ringo_warnings('confsearch', molname)

            # Save conformational pool to file
            p.descr = lambda m: f"Conformer {m.idx}"
            p.save('check.xyz')

            # Check that amide bond constraints are satisfied
            
            if 'PRDCC000211.sdf' not in start_sdf:
                for m in p:
                    for dof_idx, limits in custom_dof_limits.items():
                        atoms = [i+1 for i in dofs_list[dof_idx]]
                        dih_value = m.z(*atoms) * ringo.RAD2DEG
                        okay, info = check_dihedral_limits(dih_value, limits[0]*ringo.RAD2DEG, limits[1]*ringo.RAD2DEG)
                        if not okay:
                            raise Exception("ERROR: " + info)
                if len(manual_dihedrals) > 0:
                    for m in p:
                        assert filter_function(m), "Filter didn't work"
            
            terminate = found_appropriate(
                test_p=p,
                ref_p=confcompare_p,
                save_sdf=result_sdf
            )

    def include_either_good_or_fail(optimized_testmols_sdfs, fail_optimized_testmols_sdfs):
        assert include_if_exists(optimized_testmols_sdfs.get_path(), optimized_testmols_sdfs) or \
            include_if_exists(fail_optimized_testmols_sdfs.get_path(), fail_optimized_testmols_sdfs), \
            f"None of these exist: '{optimized_testmols_sdfs.get_path()}'/'{fail_optimized_testmols_sdfs.get_path()}'"

    bird_parser = Transformator(transformations=[
            # Create testset
            templates.extract_archive('bird_untar',
                input='testset_archive', output='testset_raw_extracted',
                aware_keys=['archivename'],
                filename=lambda original_name, archivename, **kw: f"{archivename}_{original_name.replace('_', '')}"
            ),
            templates.exec('birdcif_to_sdf',
                input='testset_raw_extracted', output='all_testmols_sdfs',
                method=birdcif_to_sdf, merged_keys=['archivename', 'ext']
            ),
            templates.copy_file('remove_noncyclic',
                input='all_testmols_sdfs', output='testmols_sdfs',
                condition=check_if_cyclic
            ),

            # Build summaries
            *primary_summary_generation_blueprint(
                grouped_transforms={
                    'get': 'summarize_full_testset',
                    'merge': 'merge_full_summaries',
                    'save': 'gen_fulltestset_summary_xlsx',
                },
                item_names={
                    'sdf': 'all_testmols_sdfs',
                    'summary': 'full_summary_object',
                    'block': 'full_summary_block',
                    'xlsx': 'full_testset_summary',
                },
            ),
            *primary_summary_generation_blueprint(
                grouped_transforms={
                    'get': 'summarize_testset',
                    'merge': 'merge_summaries',
                    'save': 'gen_testset_summary_xlsx',
                },
                item_names={
                    'sdf': 'testmols_sdfs',
                    'summary': 'summary_object',
                    'block': 'summary_block',
                    'xlsx': 'testset_summary',
                },
            ),

            # Testmol verification
            templates.map('fix_mol_topology',
                input='testmols_sdfs', output='final_testmols_sdfs',
                mapping=lambda testmols_sdfs, final_testmols_sdfs: fix_testmol_topology(input_file=testmols_sdfs, res_filename=final_testmols_sdfs)
            ),

            # templates.parse_xlsx('read_testset_summary_xlsx', input='testset_summary', output='summary_block'),
            templates.extract_df_rows('extract_summary_objects', input='summary_block', output='summary_object'),
            templates.exec('rdkit_verify_testcase',
                input=['summary_object', 'final_testmols_sdfs'], output='rdkit_status', aware_keys=['testcase'],
                method=lambda final_testmols_sdfs, rdkit_status, testcase, **kw: verify_testcase(
                    sdf=final_testmols_sdfs,
                    warning_message=rdkit_status,
                    testcase=testcase
                )
            ),
            # templates.run_gaussian('gaussian_validate',
            #     inpgeom='final_testmols_sdfs', log='gaussian_sp_logs', output='gauss_status',
            #     gjf_template=GJF_TEMPL('sp'), output_method=lambda log: get_gaussian_status(log), **GAUSSIAN_PARAMETERS
            # ),
            # templates.map('gaussian_validate_from_existing',
            #     input='gaussian_sp_logs', output='gauss_status',
            #     mapping=lambda gaussian_sp_logs: get_gaussian_status(log=gaussian_sp_logs)
            # ),

            *final_summary_generation_blueprint(
                grouped_transforms={
                    'calc_isom': 'calc_number_of_automorphisms',
                    'get': 'verify_summary_items',
                    'merge': 'merge_aug_summaries',
                    'save': 'gen_aug_testset_summary_xlsx',
                },
                item_names={
                    'sdf': 'final_testmols_sdfs',
                    'num_isom': 'num_of_automorphisms',
                    'old_summary': 'summary_object',
                    'new_summary': 'aug_summary_object',
                    'other_items': ['rdkit_status', 'gauss_status'],
                    'block': 'aug_summary_block',
                    'xlsx': 'aug_testset_summary',
                },
            ),

            # Get randomized conformers
            templates.pyfunction_subprocess('optimize_initial_testconfs',
                input='final_testmols_sdfs', output=['optimized_testmols_sdfs', 'fail_optimized_testmols_sdfs'],
                pyfunction=optimize_initial_testconfs, calcdir='calcdir',
                argv_prepare=lambda final_testmols_sdfs, optimized_testmols_sdfs, fail_optimized_testmols_sdfs, **kw:
                    (final_testmols_sdfs.access_element(), optimized_testmols_sdfs.get_path(), fail_optimized_testmols_sdfs.get_path()),
                output_process=lambda optimized_testmols_sdfs, fail_optimized_testmols_sdfs, **kw:
                    include_either_good_or_fail(optimized_testmols_sdfs, fail_optimized_testmols_sdfs)
            ),
            # templates.parse_xlsx('read_testset_aug_summary_xlsx', input='aug_testset_summary', output='aug_summary_block'),
            templates.extract_df_rows('extract_aug_summary_objects', input='aug_summary_block', output='aug_summary_object'),
            templates.pyfunction_subprocess('randomize_conformers',
                input=['optimized_testmols_sdfs', 'aug_summary_object'], output='randomized_testmols_sdfs',
                pyfunction=randomize_testconfs, calcdir='calcdir', nproc=16,
                argv_prepare=lambda optimized_testmols_sdfs, aug_summary_object, randomized_testmols_sdfs, **kw:
                    (optimized_testmols_sdfs.access_element(), aug_summary_object.access_element()['num_automorphisms'], randomized_testmols_sdfs.get_path()),
                output_process=lambda randomized_testmols_sdfs, **kw:
                    include_if_exists(randomized_testmols_sdfs.get_path(), randomized_testmols_sdfs)
            ),
        ],
        storage=ds,
        logger=main_logger,
    )
    return bird_parser
