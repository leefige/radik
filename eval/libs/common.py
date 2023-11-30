import os
import os.path

BASE_DIR = os.path.abspath("/radik")
PLOT_DIR = os.path.join(BASE_DIR, "plot")
BITONIC_BIN = os.path.join(BASE_DIR, "bitonic", "compareTopKAlgorithms")
BLOCKSELECT_BIN = os.path.join(BASE_DIR, "blockselect", "test_block_select.out")
PQ_BIN = os.path.join(BASE_DIR, "radik", "test_PQ.out")
RADIK_BIN = os.path.join(BASE_DIR, "radik", "test_radik.out")
ABLATION_BIN = os.path.join(BASE_DIR, "radik", "test_ablation.out")
