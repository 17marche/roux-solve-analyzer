"""Constants, piece definitions, and face mappings for the Rubik's Cube."""

from enum import Enum, IntEnum


class Face(IntEnum):
    U = 0
    D = 1
    F = 2
    B = 3
    L = 4
    R = 5


class Color(IntEnum):
    WHITE = 0
    YELLOW = 1
    GREEN = 2
    BLUE = 3
    ORANGE = 4
    RED = 5


class Center(IntEnum):
    U = 0
    D = 1
    F = 2
    B = 3
    L = 4
    R = 5


class Corner(IntEnum):
    """Corner pieces indexed 0..7.
    
    Coordinate representation:
    0: UFL (U=0, F=2, L=4)
    1: ULB (U=0, L=4, B=3)
    2: UBR (U=0, B=3, R=5)
    3: URF (U=0, R=5, F=2)
    4: DLF (D=1, L=4, F=2)
    5: DBL (D=1, B=3, L=4)
    6: DRB (D=1, R=5, B=3)
    7: DFR (D=1, F=2, R=5)
    """
    UFL = 0
    ULB = 1
    UBR = 2
    URF = 3
    DLF = 4
    DBL = 5
    DRB = 6
    DFR = 7


class Edge(IntEnum):
    """Edge pieces indexed 0..11.
    
    Coordinate representation:
    0: UF (U=0, F=2)
    1: UL (U=0, L=4)
    2: UB (U=0, B=3)
    3: UR (U=0, R=5)
    4: DF (D=1, F=2)
    5: DL (D=1, L=4)
    6: DB (D=1, B=3)
    7: DR (D=1, R=5)
    8: FL (F=2, L=4)
    9: BL (B=3, L=4)
    10: BR (B=3, R=5)
    11: FR (F=2, R=5)
    """
    UF = 0
    UL = 1
    UB = 2
    UR = 3
    DF = 4
    DL = 5
    DB = 6
    DR = 7
    FL = 8
    BL = 9
    BR = 10
    FR = 11


class MoveType(Enum):
    FACE = "FACE"
    SLICE = "SLICE"
    WIDE = "WIDE"
    ROTATION = "ROTATION"


# Number of piece positions and orientation modulo
NUM_CORNERS = 8
NUM_EDGES = 12
NUM_CENTERS = 6

CORNER_ORIENT_MOD = 3
EDGE_ORIENT_MOD = 2

# Facelet name strings for visual representations
CORNER_NAMES = ["UFL", "ULB", "UBR", "URF", "DLF", "DBL", "DRB", "DFR"]
EDGE_NAMES = ["UF", "UL", "UB", "UR", "DF", "DL", "DB", "DR", "FL", "BL", "BR", "FR"]
CENTER_NAMES = ["U", "D", "F", "B", "L", "R"]
