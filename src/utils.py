import sys
sys.path.append("../")
import os
import warnings

from stages import *

def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)
