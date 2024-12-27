from .registry import optimizer, find_optimizer

from . import ConvertConstantToInitializer
from . import ConvertShapeToInitializer
from . import EliminateCast
from . import EliminateConcat
from . import EliminateIdentity
from . import EliminateReshape
from . import FoldConstant
from . import OnnxSimplifier
from . import EliminateQdqPair
from . import FuseBnIntoConv
from . import FuseBnIntoGemm
