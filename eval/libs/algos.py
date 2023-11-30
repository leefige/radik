from enum import IntEnum
from collections import OrderedDict

import libs.bitonic as bitonic
import libs.blockselect as blockselect
import libs.pqs as pqs
import libs.radik as radik

class Algo(IntEnum):
    RADIK = 0
    BITONIC = 1
    PQ_GRID = 2
    PQ_BLOCK = 3

ALGOS = OrderedDict()
ALGOS[Algo.RADIK] = ("RadiK", radik.run_radik)
ALGOS[Algo.BITONIC] = ("bitonic", bitonic.run_bitonic_select)
ALGOS[Algo.PQ_GRID] = ("PQ-grid", pqs.run_grid_select)
ALGOS[Algo.PQ_BLOCK] = ("PQ-block", blockselect.run_block_select)
