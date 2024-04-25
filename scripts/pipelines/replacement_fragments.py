import networkx as nx

def get_molgraph_replacements():
    thioamidefrag = nx.Graph()
    thioamidefrag.add_node(0, symbol='N')
    thioamidefrag.add_node(1, symbol='S')
    thioamidefrag.add_node(2, symbol='C')
    thioamidefrag.add_node(3, symbol='H')
    thioamidefrag.add_node(4)
    thioamidefrag.add_edge(0, 2, type=1) # N-C
    thioamidefrag.add_edge(1, 2, type=1) # S-C
    thioamidefrag.add_edge(0, 3, type=1) # N-H
    thioamidefrag.add_edge(2, 4, type=1) # C-X
    thioamidefrag_data = {
        'name': 'thioamide',
        'subgraph': thioamidefrag,
        'charges': {
            0: 0,
            2: 0
        },
        'check_valence': [(0, 3), (2, 3)],
        'fix_bondtypes': [(0, 2, 2)],
        'remove_atoms': [3],
        # 'protect_atoms': [3],
    }
    
    nitrogoupfrag = nx.Graph()
    nitrogoupfrag.add_node(0, symbol='N')
    nitrogoupfrag.add_node(1, symbol='O')
    nitrogoupfrag.add_node(2, symbol='O')
    nitrogoupfrag.add_node(3)
    nitrogoupfrag.add_edge(0, 1, type=2) # N=O
    nitrogoupfrag.add_edge(0, 2, type=1) # N-O
    nitrogoupfrag.add_edge(0, 3, type=1) # N-X
    nitrogoupfrag_data = {
        'name': 'nitrogroup',
        'subgraph': nitrogoupfrag,
        'charges': {
            # 0: 0,
            0: +1,
            2: -1,
        },
        'check_valence': [(0, 3)]
    }
    
    badaminoacid = nx.Graph()
    badaminoacid.add_node(0, symbol='C')
    badaminoacid.add_node(1, symbol='O')
    badaminoacid.add_node(2, symbol='N')
    badaminoacid.add_node(3, symbol='H')
    badaminoacid.add_node(4)
    badaminoacid.add_node(5)
    badaminoacid.add_node(6)
    badaminoacid.add_edge(0, 1, type=1) # C-O
    badaminoacid.add_edge(0, 2, type=1) # C-N
    badaminoacid.add_edge(1, 3, type=1) # O-H
    badaminoacid.add_edge(0, 4, type=1)
    badaminoacid.add_edge(2, 5, type=1)
    badaminoacid.add_edge(2, 6, type=1)
    badaminoacid_data = {
        'name': 'badaminoacid',
        'subgraph': badaminoacid,
        'charges': {},
        'check_valence': [(0, 3), (1, 2), (2, 3)],
        'fix_bondtypes': [(0, 1, 2)],
        'remove_atoms': [3],
    }
    
    badester = nx.Graph()
    badester.add_node(0, symbol='C')
    badester.add_node(1, symbol='O')
    badester.add_node(2, symbol='O')
    badester.add_node(3, symbol='H')
    badester.add_node(4)
    badester.add_node(5)
    badester.add_edge(0, 1, type=1) # C-O(H)
    badester.add_edge(0, 2, type=1) # C-O
    badester.add_edge(1, 3, type=1) # O-H
    badester.add_edge(0, 4, type=1)
    badester.add_edge(2, 5, type=1)
    badester_data = {
        'name': 'badester',
        'subgraph': badester,
        'charges': {},
        'check_valence': [(0, 3), (1, 2), (2, 2)],
        'fix_bondtypes': [(0, 1, 2)],
        'remove_atoms': [3],
    }
    
    badaminoacidB = nx.Graph()
    badaminoacidB.add_node(0, symbol='C')
    badaminoacidB.add_node(1, symbol='O')
    badaminoacidB.add_node(2, symbol='N')
    badaminoacidB.add_node(3, symbol='H')
    badaminoacidB.add_node(4, symbol='H')
    badaminoacidB.add_node(5)
    badaminoacidB.add_node(6)
    badaminoacidB.add_edge(0, 1, type=2) # C-O
    badaminoacidB.add_edge(0, 2, type=1) # C-N
    badaminoacidB.add_edge(0, 5, type=1) # C-R
    badaminoacidB.add_edge(2, 3, type=1) # N-H
    badaminoacidB.add_edge(2, 4, type=1) # N-H
    badaminoacidB.add_edge(2, 6, type=1) # N-R
    badaminoacidB_data = {
        'name': 'badaminoacidB',
        'subgraph': badaminoacidB,
        'charges': {2: 0},
        'check_valence': [(0, 3), (1, 1), (2, 4)],
        'remove_atoms': [4],
    }
    
    oxygenposit = nx.Graph()
    oxygenposit.add_node(0, symbol='O')
    oxygenposit.add_node(1)
    oxygenposit.add_node(2)
    oxygenposit.add_edge(0, 1, type=2) # O=R1
    oxygenposit.add_edge(0, 2, type=1) # O-R2
    oxygenposit_data = {
        'name': 'oxygenposit',
        'subgraph': oxygenposit,
        'charges': {0: +1},
        'check_valence': [(0, 2)],
    }
    
    missCNdouble = nx.Graph()
    missCNdouble.add_node(0, symbol='C')
    missCNdouble.add_node(1, symbol='N')
    missCNdouble.add_node(2)
    missCNdouble.add_node(3)
    missCNdouble.add_node(4)
    missCNdouble.add_edge(0, 1, type=1) # C-N
    missCNdouble.add_edge(1, 2, type=1)
    missCNdouble.add_edge(0, 3, type=1)
    missCNdouble.add_edge(0, 4, type=1)
    missCNdouble_data = {
        'name': 'missCNdouble',
        'subgraph': missCNdouble,
        'charges': {0: 0, 1: 0},
        'check_valence': [(0, 3), (1, 2)],
        'fix_bondtypes': [(0, 1, 2)],
        'protect_atoms': [1],
    }
    
    missNNdouble = nx.Graph()
    missNNdouble.add_node(0, symbol='N')
    missNNdouble.add_node(1, symbol='N')
    missNNdouble.add_node(2)
    missNNdouble.add_node(3)
    missNNdouble.add_edge(0, 1, type=1) # N-N
    missNNdouble.add_edge(1, 2, type=1)
    missNNdouble.add_edge(0, 3, type=1)
    missNNdouble_data = {
        'name': 'missNNdouble',
        'subgraph': missNNdouble,
        'charges': {0: 0, 1: 0},
        'check_valence': [(0, 2), (1, 2)],
        'fix_bondtypes': [(0, 1, 2)],
        'protect_atoms': [1],
    }
    
    return (
        thioamidefrag_data,
        nitrogoupfrag_data,
        badaminoacid_data,
        badaminoacidB_data,
        oxygenposit_data,
        missCNdouble_data,
        missNNdouble_data,
        badester_data,
    )
