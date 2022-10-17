import sys
sys.path.append("../")
import stages
from stages import *

def str_to_class(stage, model):

    return getattr(getattr(stages, stage), model)
