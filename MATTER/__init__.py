import os, sys
import numpy as np
__version__ = '0.0.1'
project_root = '/'.join(__file__.split('/')[:-1])
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.set_printoptions(precision=6, sci_mode=False)
np.set_printoptions(precision=3, suppress=True)