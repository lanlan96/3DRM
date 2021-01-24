object_dict = {
            'non-object': 0,
            'clutter':    1,
            'wall':       2,
            'door':       3,
            'ceiling':    4,
            'floor':      5,
            'chair':      6,
            'bookcase':   7,
            'board':      8,
            'table':      9,
            'beam':       10,
            'column':     11,
            '_':          12,
            'window':     13,
            'sofa':       14
}

object_dict2 = {
            'clutter':    0,
            'wall':       1,
            'door':       2,
            'ceiling':    3,
            'floor':      4,
            'chair':      5,
            'bookcase':   6,
            'board':      7,
            'table':      8,
            'beam':       9,
            'column':     10,
            'window':     11,
            'sofa':       12
}

label2color = {
    -2: [139, 119, 101],  # negative proposal
    0: [50, 50, 50],  # clutter 323232
    1: [115, 252, 253],  # wall 73fcfd
    2: [201, 199, 114],  # door c9c772
    3: [117,250,76],  # ceiling 75fa4c
    4: [0,35,245],  # floor, 0023f5
    5: [235, 51, 35],  # chair eb3323
    6: [91,196,111],  # bookcase 5bc46f
    7: [200,200,200],  # board c8c8c8
    8: [163, 124, 195],  # table a37cc3
    9: [255,253,85],  # beam fffd55
    10: [234,63,247],  #column ea3ff7
    11: [98,106,246],  # window 626af6
    12: [188,106,103],  # sofa bc6a67
}