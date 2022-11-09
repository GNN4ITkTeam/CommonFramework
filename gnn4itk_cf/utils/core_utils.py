from gnn4itk_cf import stages
from gnn4itk_cf.stages import *

def str_to_class(stage, model):
    """
    Convert a string to a class in the stages directory
    """
    
    return getattr(getattr(stages, stage), model)
