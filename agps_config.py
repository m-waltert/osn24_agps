import pandas as pd
from shapely.geometry import Polygon

# %%
# Only consider the following typecodes
AC2CONSIDER = [
    "A20N",
    "A21N",
    "A319",
    "A320",
    "A321",
    "A332",
    "A333",
    "A343",
    "A359",
    "A35K",
    "A388",
    "B38M",
    "B39M",
    "B733",
    "B734",
    "B735",
    "B736",
    "B737",
    "B738",
    "B739",
    "B744",
    "B752",
    "B753",
    "B762",
    "B763",
    "B764",
    "B772",
    "B773",
    "B77L",
    "B77W",
    "B788",
    "B789",
    "B78X",
    "BCS1",
    "BCS3",
    "CRJ2",
    "CRJ7",
    "CRJ9",
    "CRJX",
    "E190",
    "E195",
    "E290",
    "E295",
    "E75L",
    "E75S",
]

# These aircraft typecodes are considered "older types"
OLD_AIRCRAFT = [
    "B733",
    "B733",
    "B734",
    "B735",
    "B744",
    "B772",
    "B762",
    "B763",
    "B764",
    "B752",
    "B753",
    "A343",
]


# %%
# Information on aircraft types not contained in openap.prop
# Data is based on https://contentzone.eurocontrol.int/aircraftperformance
AIRCRAFT_INFO = {
    "BCS1": {"max_pax": 110, "engine": "PW1524G", "n_engines": 2},
    "BCS3": {"max_pax": 130, "engine": "PW1524G", "n_engines": 2},
    "E290": {"max_pax": 120, "engine": "PW1919G", "n_engines": 2},
    "E295": {"max_pax": 132, "engine": "PW1921G", "n_engines": 2},
    "A35K": {"max_pax": 350, "engine": "trent xwb-97", "n_engines": 2},
    "CRJ9": {"max_pax": 90, "engine": "CF34-8C5", "n_engines": 2},
    "B77L": {"max_pax": 320, "engine": "Trent 892", "n_engines": 2},
    "B78X": {"max_pax": 300, "engine": "Trent 1000-K2", "n_engines": 2},
    "B733": {"max_pax": 130, "engine": "CFM56-3B", "n_engines": 2},
    "B735": {"max_pax": 108, "engine": "CFM56-3B", "n_engines": 2},
    "B736": {"max_pax": 108, "engine": "CFM56-7B", "n_engines": 2},
    "B753": {"max_pax": 243, "engine": "PW2037", "n_engines": 2},
    "B762": {"max_pax": 210, "engine": "CF6-80C2B7F", "n_engines": 2},
    "B764": {"max_pax": 245, "engine": "CF6-80C2B8F", "n_engines": 2},
    "CRJ2": {"max_pax": 50, "engine": "CF34", "n_engines": 2},
    "CRJ7": {"max_pax": 70, "engine": "CF34-8C1", "n_engines": 2},
    "CRJ9": {"max_pax": 90, "engine": "CF34-8C5", "n_engines": 2},
    "CRJX": {"max_pax": 104, "engine": "CF34-8C5A1", "n_engines": 2},
    "E75S": {"max_pax": 78, "engine": "CF34-8E6", "n_engines": 2},
}

# %%
# Engine Start Default Parameters:
DEFAULT_STARTUP_TIME = 60  # Duration of start-up of an engine [seconds]
DEFAULT_WARMUP_TIME = (
    120  # Duration of warm-up period required after start-up of last engine [seconds]
)

# Default Pushback Tug parameters
DEFAULT_SFC_AGPS = (
    20 / 3600
)  # kg/h -> kg/s (from Postorino_etal_2019), other source: Deonandan_Balakrishnan_2010
DEFAULT_SPEED_AGPS = 15  # Driving speed of tug [km/h]
DEFAULT_BUFFER_AGPS = pd.Timedelta(
    minutes=15
)  # Buffer time between two consecutive pushbacks (of one single AGPS unit)

# Driving distances from Runway to Stand Areas in kilometers [km]
# All runways to unknwon: 2.8km (which is equal to the average of the distances to the other stand areas)
DISTANCE_MATRIX = pd.DataFrame(
    data=[  # A_N    AB_C    B_S     C,      D,      E,      F,      G,      H,      I,      T,      GAC     unknown
        [4.4, 3.8, 3.7, 3.5, 3.3, 5.3, 4.6, 2.8, 4.4, 4.7, 3.3, 5.0, 2.8],  # RWY 10
        [3.6, 3.6, 3.8, 4.4, 4.2, 2.1, 3.7, 5.1, 3.6, 3.9, 4.6, 4.1, 2.8],  # RWY 16
        [1.3, 1.5, 1.5, 2.0, 1.8, 2.0, 0.6, 2.7, 1.2, 1.5, 2.2, 0.4, 2.8],  # RWY 28
        [2.4, 2.5, 2.7, 3.3, 3.0, 1.1, 1.8, 3.8, 2.4, 2.7, 3.3, 1.6, 2.8],  # RWY 32
        [2.2, 1.7, 1.4, 1.3, 1.0, 3.0, 2.4, 0.5, 2.2, 2.4, 1.0, 2.7, 2.8],  # RWY 34
    ],
    index=["10", "16", "28", "32", "34"],  # Takeoff runways as rows
    columns=[
        "A North",
        "AB Courtyard",
        "B South",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "T",
        "GAC",
        "unknown",
    ],  # Stand Areas as columns
)

# %%
# APU Fuel Flow Data, based on ICAO Doc 9889 Airport Air Quality Manual, Table 3-A1-6. APU fuel group
data = {
    "APU fuel group": [
        "Business jets/regional jets (seats < 100)",
        "Smaller (100 ≤ seats < 200), newer types",
        "Smaller (100 ≤ seats < 200), older types",
        "Mid-range (200 ≤ seats < 300), all types",
        "Larger (300 ≤ seats), older types",
        "Larger (300 ≤ seats), newer types",
    ],
    "startup": [68, 77, 69, 108, 106, 146],  # Start-up No load (kg/h)
    "normal": [101, 110, 122, 164, 202, 238],  # Normal running Maximum ECS (kg/h)
    "high": [110, 130, 130, 191, 214, 262],  # High load Main engine start (kg/h)
}

DF_APU = pd.DataFrame(data)


# %%
# Nose-in push out stands at ZRH
def get_PushStands_LSZH():
    stands = []

    # Dock E Stands
    stands.append(
        Polygon(
            shell=(
                (8.549924, 47.462661),
                (8.551635, 47.460186),
                (8.559250, 47.459671),
                (8.559618, 47.462015),
            )
        )
    )

    # Dock A Stands
    stands.append(
        Polygon(
            shell=(
                (8.555001, 47.454855),
                (8.556140, 47.453175),
                (8.560965, 47.452810),
                (8.562083, 47.454330),
            )
        )
    )

    # Dock B Stands
    stands.append(
        Polygon(
            shell=(
                (8.556163, 47.452404),
                (8.558079, 47.449825),
                (8.559995, 47.449686),
                (8.560519, 47.452131),
            )
        )
    )

    # Charlie Stands
    stands.append(
        Polygon(
            shell=(
                (8.560601, 47.448472),
                (8.562883, 47.445380),
                (8.563924, 47.445758),
                (8.561669, 47.448833),
            )
        )
    )

    # Golf Stands (east part)
    stands.append(
        Polygon(
            shell=(
                (8.562589, 47.440829),
                (8.561054, 47.443088),
                (8.560208, 47.442844),
                (8.561806, 47.440590),
            )
        )
    )

    # Golf Stands (west part)
    stands.append(
        Polygon(
            shell=(
                (8.563977, 47.441257),
                (8.563025, 47.442721),
                (8.562171, 47.442462),
                (8.563191, 47.441011),
            )
        )
    )

    # Papa Stands
    stands.append(
        Polygon(
            shell=(
                (8.551049, 47.463711),
                (8.550944, 47.463068),
                (8.553608, 47.462883),
                (8.553666, 47.463552),
            )
        )
    )

    # Tango Stands
    stands.append(
        Polygon(
            shell=(
                (8.561366, 47.443951),
                (8.562294, 47.443334),
                (8.563789, 47.443741),
                (8.563284, 47.444532),
            )
        )
    )

    # Whiskey Stands
    stands.append(
        Polygon(
            shell=(
                (8.546829, 47.454721),
                (8.545534, 47.454426),
                (8.547307, 47.452003),
                (8.548511, 47.452336),
            )
        )
    )

    return stands


def get_Stands_LSZH():
    stands = {
        # Dock E Stands
        "E": Polygon(
            [
                (8.546718, 47.464071),
                (8.550259, 47.464869),
                (8.559958, 47.462345),
                (8.559374, 47.458461),
                (8.550156, 47.459276),
            ]
        ),
        # India Stands
        "I": Polygon(
            [
                (8.552900, 47.457306),
                (8.558658, 47.456916),
                (8.558383, 47.455301),
                (8.553816, 47.455573),
            ]
        ),
        # Hotel Stands
        "H": Polygon(
            [
                (8.558658, 47.456916),
                (8.562658, 47.456421),
                (8.562465, 47.455041),
                (8.558383, 47.455301),
            ]
        ),
        # Foxtrott Stands
        "F": Polygon(
            [
                (8.564507, 47.455653),
                (8.570081, 47.455319),
                (8.569989, 47.454329),
                (8.565184, 47.454873),
            ]
        ),
        # GAC Stands
        "GAC": Polygon(
            [
                (8.571513, 47.455682),
                (8.576299, 47.455459),
                (8.576045, 47.453341),
                (8.571947, 47.453482),
            ]
        ),
        # A North Stands
        "A North": Polygon(
            [
                (8.554012, 47.455346),
                (8.562447, 47.454739),
                (8.561954, 47.453596),
                (8.554955, 47.454031),
            ]
        ),
        # A South/B North Stands
        "AB Courtyard": Polygon(
            [
                (8.554955, 47.454031),
                (8.561954, 47.453596),
                (8.560647, 47.451033),
                (8.556964, 47.451245),
            ]
        ),
        # B South Stands
        "B South": Polygon(
            [
                (8.556964, 47.451245),
                (8.560647, 47.451033),
                (8.560361, 47.449203),
                (8.558008, 47.449326),
            ]
        ),
        # Charlie Stands
        "C": Polygon(
            [
                (8.557030, 47.449262),
                (8.559435, 47.449255),
                (8.562857, 47.444292),
                (8.560348, 47.443866),
            ]
        ),
        # Delta Stands
        "D": Polygon(
            [
                (8.559435, 47.449255),
                (8.561566, 47.449141),
                (8.564432, 47.445080),
                (8.562857, 47.444292),
            ]
        ),
        # Golf Stands
        "G": Polygon(
            [
                (8.559520, 47.443254),
                (8.563244, 47.443221),
                (8.564747, 47.440958),
                (8.561925, 47.439725),
            ]
        ),
        # Tango Stands
        "T": Polygon(
            [
                (8.564432, 47.445080),
                (8.563445, 47.446635),
                (8.567162, 47.447636),
                (8.568790, 47.442709),
                (8.564747, 47.440958),
                (8.563244, 47.443221),
            ]
        ),
    }

    return stands


# %%
# icao24 hex code often observed at LSZH Airport, but currently missing in Opensky Database
data = [
    ("48ac80", "E295", "L2J"),
    ("502d7a", "BCS3", "L2J"),
    ("4b193a", "A320", "L2J"),
    ("4b194c", "A21N", "L2J"),
    ("502d95", "BCS3", "L2J"),
    ("502d88", "BCS3", "L2J"),
    ("4b028b", "E290", "L2J"),
    ("4b0291", "E290", "L2J"),
    ("4b0290", "E290", "L2J"),
    ("4b0292", "E290", "L2J"),
    ("4b0293", "E295", "L2J"),
    ("4b028c", "E290", "L2J"),
    ("4b028d", "E290", "L2J"),
    ("4864ea", "E295", "L2J"),
    ("486482", "E295", "L2J"),
    ("4b0294", "E295", "L2J"),
    ("4b0295", "E295", "L2J"),
    ("4b1818", "A20N", "L2J"),
    ("502d7c", "BCS3", "L2J"),
    ("39e690", "BCS3", "L2J"),
    ("05a0a6", "A319", "L2J"),
    ("4c01f2", "A320", "L2J"),
    ("4bc88c", "A21N", "L2J"),
    ("4b19fd", "E190", "L2J"),
    ("4b19e6", "E195", "L2J"),
    ("4b19ef", "E195", "L2J"),
    ("4b19fc", "E195", "L2J"),
    ("4b19ff", "E195", "L2J"),
    ("4b19fe", "E190", "L2J"),
    ("4b194d", "A21N", "L2J"),
    ("4b1942", "A320", "L2J"),
    ("502d7d", "BCS3", "L2J"),
    ("4b3778", "PC24", "L2J"),
    ("4b0dd0", "PC12", "L1T"),
    ("3c75a5", "E145", "L2J"),
    ("440c6f", "A21N", "L2J"),
    ("4cadea", "A20N", "L2J"),
    ("4d2507", "A320", "L2J"),
    ("494119", "E55P", "L2J"),
    ("4b19c0", "FA6X", "L2J"),
    ("4d2494", "A320", "L2J"),
    ("4cade8", "A20N", "L2J"),
    ("4d24ba", "A20N", "L2J"),
    ("4401d4", "E55P", "L2J"),
    ("4d2411", "A310", "L2J"),
    ("502d5f", "BCS3", "L2J"),
    ("4b193f", "A320", "L2J"),
    ("502d5d", "A320", "L2J"),
    ("4b3810", "PC24", "L2J"),
    ("440da4", "C56X", "L2J"),
    ("452135", "A320", "L2J"),
    ("4d24de", "A20N", "L2J"),
    ("4b1f30", "DA42", "L2T"),
    ("4b0e4a", "PC12", "L1T"),
    ("4b37b7", "SF50", "L1J"),
    ("502d89", "BCS3", "L2J"),
    ("39bda0", "BCS3", "L2J"),
    ("39e697", "BCS3", "L2J"),
    ("46b8a8", "A20N", "L2J"),
    ("39e699", "BCS3", "L2J"),
    ("502d5e", "BCS3", "L2J"),
    ("39bda6", "BCS3", "L2J"),
    ("3c4dc9", "CRJ9", "L2J"),
    ("46b8b0", "A20N", "L2J"),
    ("46b8ab", "A20N", "L2J"),
    ("46b8b2", "A20N", "L2J"),
    ("4bb1ed", "A332", "L2J"),
    ("4b18d9", "CL35", "L2J"),
    ("39bdaa", "BCS3", "L2J"),
    ("46b8b1", "A20N", "L2J"),
]


# Convert to a DataFrame
DF_MISSING_ICAO24 = pd.DataFrame(
    data, columns=["icao24", "typecode", "icaoaircrafttype"]
)
