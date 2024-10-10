import pandas as pd
from shapely.geometry import Polygon

#%%
# Only consider the following typecodes
AC2CONSIDER = ['A20N', 'A21N', 'A319', 'A320', 'A321', 'A332', 'A333', 'A343', 'A359', 'A35K', 'A388', 'B38M',
               'B39M', 'B733', 'B734', 'B735', 'B736', 'B737', 'B738', 'B739', 'B744', 'B752', 'B753', 'B762', 
               'B763', 'B764', 'B772', 'B773', 'B77L', 'B77W', 'B788', 'B789', 'B78X', 'BCS1', 'BCS3', 'CRJ2', 
               'CRJ7', 'CRJ9', 'CRJX', 'E190', 'E195', 'E290', 'E295', 'E75L', 'E75S']

# These aircraft typecodes are considered "older types"
OLD_AIRCRAFT = ['B733', 'B733', 'B734', 'B735', 'B744', 'B772', 'B762', 'B763', 'B764', 'B752', 'B753', 'A343']


#%%
# Information on aircraft types not contained in openap.prop
# Data is based on https://contentzone.eurocontrol.int/aircraftperformance
AIRCRAFT_INFO = {
    'BCS1': {
        'max_pax': 110,
        'engine': 'PW1524G',
        'n_engines': 2
    },
    'BCS3': {
        'max_pax': 130,
        'engine': 'PW1524G',
        'n_engines': 2
    },
    'E290': {
        'max_pax': 120,
        'engine': 'PW1919G',
        'n_engines': 2
    },
    'E295': {
        'max_pax': 132,
        'engine': 'PW1921G',
        'n_engines': 2
    },
    'A35K': {
        'max_pax': 350,
        'engine': 'trent xwb-97',
        'n_engines': 2
    },
    'CRJ9': {
        'max_pax': 90,
        'engine': 'CF34-8C5',
        'n_engines': 2
    },
    'B77L': {
        'max_pax': 320,
        'engine': 'Trent 892',
        'n_engines': 2
    },
    'B78X': {
        'max_pax': 300,
        'engine': 'Trent 1000-K2',
        'n_engines': 2
    },
    'B733': {
        'max_pax': 130,
        'engine': 'CFM56-3B',
        'n_engines': 2
    },
    'B735': {
        'max_pax': 108,
        'engine': 'CFM56-3B',
        'n_engines': 2
    },
    'B736': {
        'max_pax': 108,
        'engine': 'CFM56-7B',
        'n_engines': 2
    },
    'B753': {
        'max_pax': 243,
        'engine': 'PW2037',
        'n_engines': 2
    },
    'B762': {
        'max_pax': 210,
        'engine': 'CF6-80C2B7F',
        'n_engines': 2
    },
    'B764': {
        'max_pax': 245,
        'engine': 'CF6-80C2B8F',
        'n_engines': 2
    },
    'CRJ2': {
        'max_pax': 50,
        'engine': 'CF34',
        'n_engines': 2
    },
    'CRJ7': {
        'max_pax': 70,
        'engine': 'CF34-8C1',
        'n_engines': 2
    },
    'CRJ9': {
        'max_pax': 90,
        'engine': 'CF34-8C5',
        'n_engines': 2
    },
    'CRJX': {
        'max_pax': 104,
        'engine': 'CF34-8C5A1',
        'n_engines': 2
    },
    'E75S': {
        'max_pax': 78,
        'engine': 'CF34-8E6',
        'n_engines': 2
    },
}

#%%
# Engine Start Default Parameters:
DEFAULT_STARTUP_TIME = 60                # Duration of start-up of an engine [seconds]
DEFAULT_WARMUP_TIME = 120                # Duration of warm-up period required after start-up of last engine [seconds]

#%% 


# APU Fuel Flow Data, based on ICAO Doc 9889 Airport Air Quality Manual, Table 3-A1-6. APU fuel group 
data = {
    'APU fuel group': [
        'Business jets/regional jets (seats < 100)',
        'Smaller (100 ≤ seats < 200), newer types',
        'Smaller (100 ≤ seats < 200), older types',
        'Mid-range (200 ≤ seats < 300), all types',
        'Larger (300 ≤ seats), older types',
        'Larger (300 ≤ seats), newer types'
    ],
'startup': [68, 77, 69, 108, 106, 146],                 # Start-up No load (kg/h)
'normal': [101, 110, 122, 164, 202, 238],               # Normal running Maximum ECS (kg/h)
'high': [110, 130, 130, 191, 214, 262]                 # High load Main engine start (kg/h)
}

DF_APU = pd.DataFrame(data)


def get_Stands_LSZH():

    stands = []

    # # Dock E Stands
    # stands.append(Polygon(shell=((8.559361, 47.461887),
    #                              (8.549978, 47.462542),
    #                              (8.551437, 47.460347),
    #                              (8.559075, 47.459822))))


    # Dock E Stands (area slightly extended in order to improve pushback detection)
    stands.append(Polygon(shell=((8.549924, 47.462661),
                                (8.551635, 47.460186),
                                (8.559250, 47.459671),
                                (8.559618, 47.462015))))

    # Dock A Stands
    stands.append(Polygon(shell=((8.555001, 47.454855),
                                (8.556140, 47.453175),
                                (8.560965, 47.452810),
                                (8.562083, 47.454330))))

    # # Dock B Stands
    # stands.append(Polygon(shell=((8.560488, 47.451963),
    #                              (8.556346, 47.452250),
    #                              (8.557920, 47.450010),
    #                              (8.560157, 47.449799))))

    # Dock B Stands (area slightly extended in order to improve pushback detection)
    stands.append(Polygon(shell=((8.556163, 47.452404),
                                (8.558079, 47.449825),
                                (8.559995, 47.449686),
                                (8.560519, 47.452131))))

    # Charlie Stands
    stands.append(Polygon(shell=((8.560601, 47.448472),
                                (8.562883, 47.445380),
                                (8.563924, 47.445758),
                                (8.561669, 47.448833))))

    # Golf Stands (east part)
    stands.append(Polygon(shell=((8.562589, 47.440829),
                                (8.561054, 47.443088),
                                (8.560208, 47.442844),
                                (8.561806, 47.440590))))

    # Golf Stands (west part)
    stands.append(Polygon(shell=((8.563977, 47.441257),
                                (8.563025, 47.442721),
                                (8.562171, 47.442462),
                                (8.563191, 47.441011))))

    # Papa Stands
    stands.append(Polygon(shell=((8.551049, 47.463711),
                                (8.550944, 47.463068),
                                (8.553608, 47.462883),
                                (8.553666, 47.463552))))

    # Tango Stands
    stands.append(Polygon(shell=((8.561366, 47.443951),
                                (8.562294, 47.443334),
                                (8.563789, 47.443741),
                                (8.563284, 47.444532))))

    # Whiskey Stands
    stands.append(Polygon(shell=((8.546829, 47.454721),
                                (8.545534, 47.454426),
                                (8.547307, 47.452003),
                                (8.548511, 47.452336))))


    return stands


# icao24 missing in Opensky Database
data = [
        ('48ac80', 'E295', 'L2J'),
        ('502d7a', 'BCS3', 'L2J'),
        ('4b193a', 'A320', 'L2J'),
        ('4b194c', 'A21N', 'L2J'),
        ('502d95' ,'BCS3', 'L2J'),
        ('502d88', 'BCS3', 'L2J'),
        ('4b028b', 'E290', 'L2J'),
        ('4b0291', 'E290', 'L2J'),
        ('4b0290', 'E290', 'L2J'),
        ('4b0292', 'E290', 'L2J'),
        ('4b0293', 'E295', 'L2J'),
        ('4b028c', 'E290', 'L2J'),
        ('4b028d', 'E290', 'L2J'),
        ('4864ea', 'E295', 'L2J'),
        ('486482', 'E295', 'L2J'),
        ('4b0294', 'E295', 'L2J'),
        ('4b0295', 'E295', 'L2J'),
        ('4b1818', 'A20N', 'L2J'),
        ('502d7c', 'BCS3', 'L2J'),
        ('39e690', 'BCS3', 'L2J'),
        ('05a0a6', 'A319', 'L2J'),
        ('4c01f2', 'A320', 'L2J'),
        ('4bc88c', 'A21N', 'L2J'),
        ('4b19fd', 'E190', 'L2J'),
        ('4b19e6', 'E195', 'L2J'),
        ('4b19ef', 'E195', 'L2J'),
        ('4b19fc', 'E195', 'L2J'),
        ('4b19ff', 'E195', 'L2J'),
        ('4b19fe', 'E190', 'L2J'),
        ('4b194d', 'A21N', 'L2J'),
        ('4b1942', 'A320', 'L2J'),
        ('502d7d', 'BCS3', 'L2J'),
        ('4b3778', 'PC24', 'L2J'),
        ('4b0dd0', 'PC12', 'L1T'),
        ('3c75a5', 'E145', 'L2J'),
        ('440c6f', 'A21N', 'L2J'),
        ('4cadea', 'A20N', 'L2J'),
        ('4d2507', 'A320', 'L2J'),
        ('494119', 'E55P', 'L2J'),
        ('4b19c0', 'FA6X', 'L2J'),
        ('4d2494', 'A320', 'L2J'),
        ('4cade8', 'A20N', 'L2J'),
        ('4d24ba', 'A20N', 'L2J'),
        ('4401d4', 'E55P', 'L2J'),
        ('4d2411', 'A310', 'L2J'),
        ('502d5f', 'BCS3', 'L2J'),
        ('4b193f', 'A320', 'L2J'),
        ('502d5d', 'A320', 'L2J'),
        ('4b3810' ,'PC24', 'L2J'),
        ('440da4', 'C56X', 'L2J'),
        ('452135', 'A320', 'L2J'),
        ('4d24de', 'A20N', 'L2J'),
        ('4b1f30', 'DA42', 'L2T'),
        ('4b0e4a', 'PC12', 'L1T'),
        ('4b37b7', 'SF50', 'L1J'),
        ('502d89', 'BCS3', 'L2J'),
        ('39bda0', 'BCS3', 'L2J'),
        ('39e697', 'BCS3', 'L2J'),
        ('46b8a8', 'A20N', 'L2J'),
        ('39e699', 'BCS3', 'L2J'),
        ('502d5e', 'BCS3', 'L2J'),
        ('39bda6', 'BCS3', 'L2J'),
        ('3c4dc9', 'CRJ9', 'L2J'),
        ('46b8b0', 'A20N', 'L2J'),
        ('46b8ab', 'A20N', 'L2J'),
        ('46b8b2', 'A20N', 'L2J'),
        ('4bb1ed', 'A332', 'L2J'),
        ('4b18d9', 'CL35', 'L2J'),
        ('39bdaa', 'BCS3', 'L2J'),
        ('46b8b1', 'A20N', 'L2J'),
    ]


# Convert to a DataFrame
DF_MISSING_ICAO24 = pd.DataFrame(data, columns=['icao24', 'typecode', 'icaoaircrafttype'])