import os
import os.path

from enum import IntEnum

BASE_DIR = os.path.abspath("/radik")
PLOT_DIR = os.path.join(BASE_DIR, "plot")
BITONIC_BIN = os.path.join(BASE_DIR, "bitonic", "compareTopKAlgorithms")
BLOCKSELECT_BIN = os.path.join(BASE_DIR, "blockselect", "test_block_select.out")
PQ_BIN = os.path.join(BASE_DIR, "radik", "test_PQ.out")
RADIK_BIN = os.path.join(BASE_DIR, "radik", "test_radik.out")
ABLATION_BIN = os.path.join(BASE_DIR, "radik", "test_ablation.out")

class Distribution(IntEnum):
    U_0_1 = 0
    U_0_6_0_7 = 1
    U_128_6_128_7 = 2
    ZIPF_1_1 = 3
    ALL_ZERO = 4
