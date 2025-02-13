from ..groups import GROUPS
from ..lib import Lib
from .optimizers import OPTIMIZERS


# ---------------------------------------------------------------------------- #
#                                SCIPY OPTIMIZE                                #
# ---------------------------------------------------------------------------- #
class LibScipyOptimize(Lib):
    def __init__(self):
        super().__init__(
            names = ('scipy.optimize', 'scipy', 'so',),
            requires = 'scipy.optimize',
        )

    def initialize(self):
        from ...integrations.scipy import (ScipyBasinhopping, ScipyBrute,
                                           ScipyDE, ScipyDIRECT,
                                           ScipyDualAnnealing,
                                           ScipyLeastSquares,
                                           ScipyLevenbergMarquardt,
                                           ScipyMinimize, ScipyMinimizeScalar,
                                           ScipyRoot, ScipyRootScalar,
                                           ScipySHGO)

        for method in ('nelder-mead','powell','cg','bfgs','l-bfgs-b','tnc','cobyla','cobyqa','slsqp','trust-constr',):
            self.register(ScipyMinimize.configured(method=method).set_name([f'minimize.{method}', method, ]))

        for method in ('cg','bfgs','l-bfgs-b', 'cobyla', 'cobyqa', 'slsqp'):
            self.register(ScipyMinimize.conf_rescaled(0.1, method=method).set_name([f'minimize.{method}-big', f'{method}-big', ]))
            self.register(ScipyMinimize.conf_rescaled(0.01, method=method).set_name([f'minimize.{method}-large', f'{method}-large', ]))
            self.register(ScipyMinimize.conf_rescaled(10, method=method).set_name([f'minimize.{method}-small', f'{method}-small', ]))
            self.register(ScipyMinimize.conf_rescaled(100, method=method).set_name([f'minimize.{method}-tiny', f'{method}-tiny', ]))

        self.register(ScipyDualAnnealing)
        self.register(ScipyBasinhopping)
        self.register(ScipyDE)
        for de_strat in ('best1exp', 'rand1bin', 'rand1exp', 'rand2bin', 'rand2exp', 'randtobest1bin', 'randtobest1exp', 'currenttobest1bin', 'currenttobest1exp', 'best2exp', 'best2bin'):
            self.register(
                ScipyDE.configured(strategy=de_strat).set_name((f'Differential evolution - {de_strat}', f'DE-{de_strat}')),
            )

        self.register(ScipySHGO, maxdims = 10)
        self.register(ScipySHGO.configured(n=10).set_name('SHGO-10'), maxdims = 10)
        self.register(ScipySHGO.configured(n=101).set_name('SHGO-101'), maxdims = 10)
        self.register(ScipySHGO.configured(sampling_method='halton').set_name('SHGO-halton'), maxdims = 10)
        self.register(ScipySHGO.configured(sampling_method='sobol').set_name('SHGO-sobol'), maxdims = 10)

        self.register(ScipyDIRECT, groups = (GROUPS.NO_FORCE_STOP,))

        self.register(ScipyBrute, maxdims = 3)
        self.register(ScipyBrute.configured(Ns = 10).set_name('Brute10'), maxdims = 4)
        self.register(ScipyBrute.configured(Ns = 5).set_name('Brute5'), maxdims = 8)
        self.register(ScipyBrute.configured(Ns = 3).set_name('Brute3'), maxdims = 16)

        self.register(ScipyLeastSquares.configured(method='trf').set_name(['least_squares.trf', 'trf']))
        self.register(ScipyLeastSquares.configured(method='dogbox').set_name(['least_squares.dogbox', 'dogbox']))

        self.register(ScipyLevenbergMarquardt)

        for method in ("hybr","lm","broyden1","broyden2","anderson","linearmixing","diagbroyden","excitingmixing","krylov","df-sane",):
            self.register(ScipyRoot.configured(method=method).set_name([f'root.{method}', method, ]))

        self.register(ScipyRoot.configured(method='hybr', random_weights=False).set_name(['root.hybr-norand', 'hybr-norand', ]))

        for mode in ('coord', 'mask', 'weighted'):
            for method in ('Brent', "Bounded", "Golden"):
                self.register(ScipyMinimizeScalar.configured(method = method, mode = mode).set_name([f'minimize_scalar.{method}-{mode}', f'{method}-{mode}', ]))

            for method in ('newton', "secant",):
                self.register(ScipyRootScalar.configured(method = method, mode = mode).set_name([f'root_scalar.{method}-{mode}', f'{method}-{mode}', ]))

lib_scipy = LibScipyOptimize()
OPTIMIZERS.register_lib(lib_scipy)