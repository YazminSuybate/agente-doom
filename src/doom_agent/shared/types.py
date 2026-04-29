from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

PathLike: TypeAlias = str | Path
Observation: TypeAlias = NDArray[np.uint8]
BinaryAction: TypeAlias = NDArray[np.int32]
