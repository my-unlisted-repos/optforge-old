from .optimizers import OPTIMIZERS
from ..lib import Lib

class LibPySORS(Lib):
    def __init__(self):
        super().__init__(
            names = ('pysors', 'ps'),
            requires = 'pysors',
        )

    def initialize(self):
        import pysors
        from ...integrations.pysors.pysors_optimizer import METHODS_LIST, PySORSOptimizer, RHO
        for method in METHODS_LIST:
            self.register(PySORSOptimizer.configured(optimizer=method).set_name(method)) # type:ignore

        # hparam search
        # transformer design
        self.register(PySORSOptimizer.configured(
            optimizer=pysors.BDS(
                a_init=0.6224080155867018,
                a_max=37.78499871250205,
                theta = 0.5011434436330786,
                gamma = 1.204381288663737,
                rho = RHO(4.879426594753578, 7.9194419829811435)
            )
        ).set_name('BDS-hpsearch-TransformerDesign')) # type:ignore

        self.register(PySORSOptimizer.configured(
            optimizer=pysors.BDS(
                a_init=0.0052695119041272775,
                a_max=17.438960740194638,
                theta = 0.40380041566374825,
                gamma = 1.270558486511026,
            )
        ).set_name('BDS-hpsearch-TransformerDesign-norho')) # type:ignore

        self.register(PySORSOptimizer.configured(
            optimizer=pysors.AHDS(
                a_init=0.17741091693308597,
                a_max=28.02303982377021,
                theta = 0.16462617829361914,
                gamma = 1.8870004748352192,
                rho = RHO(6.374055733283804, 9.49169601301065)
            )
        ).set_name('AHDS-hpsearch-TransformerDesign')) # type:ignore

        self.register(PySORSOptimizer.configured(
            optimizer=pysors.AHDS(
                a_init=0.6243228925010866,
                a_max=0.0642543482556448,
                theta = 0.31660531882966136,
                gamma = 9.699443797868103,
            )
        ).set_name('AHDS-hpsearch-TransformerDesign-norho')) # type:ignore

        self.register(PySORSOptimizer.configured(
            optimizer=pysors.AHDS(
                a_init=0.17489795443877743,
                a_max=0.2563801949703617,
                theta = 0.7679861773782415,
                gamma = 4.212266961859332,
                rho = RHO(13.302414222192645, 7.785573070501136)
            )
        ).set_name('AHDS-hpsearch-mixv1')) # type:ignore

        _KWARGS = {
            "a_init": 1.3916738990058146,
            "c_init": 0.8072387805967745,
            "beta": 5.033335921097635,
            "sigma_1": 0.24391634488641767,
            "sigma_2": 9.994086493906902,
            "distribution": "Normal",
            "step_upd": "half",
            "theta": 2.3706393959382197,
            "T_half": 633,
            "T_power": 1,
        }
        self.register(PySORSOptimizer.configured(
            optimizer=pysors.RSPI_FD(**_KWARGS)
        ).set_name('RSPI_FD-hpsearch-mixv1')) # type:ignore


lib_pysors = LibPySORS()
OPTIMIZERS.register_lib(lib_pysors)