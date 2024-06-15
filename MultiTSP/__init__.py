try:
    from ga_mtsp import GAMultiTSP
    from aco_mtsp import ACOMultiTSP
    from cvxpy_mtsp import CVXPYMultiTSP
    from hill_mtsp import HillClimbMultiTSP
    from tabu_mtsp import TabuSearchMultiTSP

    from utils import *
except ImportError:
    from MultiTSP.ga_mtsp import GAMultiTSP
    from MultiTSP.aco_mtsp import ACOMultiTSP
    from MultiTSP.cvxpy_mtsp import CVXPYMultiTSP
    from MultiTSP.hill_mtsp import HillClimbMultiTSP
    from MultiTSP.tabu_mtsp import TabuSearchMultiTSP

    from MultiTSP.utils import *