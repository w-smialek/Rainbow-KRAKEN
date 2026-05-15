import numpy as np
from pathlib import Path
import sys
# Allow imports from the repository root when this script is run directly.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import rkraken as rk

arr = np.load('./parameter_scan/scan_full.npy')

rk.plot_mat(arr,saveloc='aa.png')