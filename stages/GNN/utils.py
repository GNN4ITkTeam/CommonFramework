import sys

from .models.interaction_gnn import InteractionGNN

def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)