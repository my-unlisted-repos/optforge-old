
from collections import UserDict
from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, Any

if TYPE_CHECKING:
    from .optimizer import Optimizer

ConfigLiterals = Literal[
            'supports_ask',
            'supports_multiple_asks',
            'requires_batch_mode',
            'store_paramdicts',
        ]

class Config(UserDict[ConfigLiterals, Any]):
    def __init__(
        self,
        supports_ask: bool,
        supports_multiple_asks:bool,
        requires_batch_mode: bool,
        store_paramdicts: Literal['none', 'best', 'all'] = 'none',
        ):
        self.CONFIG: dict[ConfigLiterals, Any] = {
            "supports_ask": supports_ask,
            "supports_multiple_asks": supports_multiple_asks,
            "requires_batch_mode": requires_batch_mode,
            "store_paramdicts": store_paramdicts,
        }
        super().__init__(self.CONFIG) # type:ignore

    def copy(self):
        return Config(
            supports_ask = self.CONFIG['supports_ask'],
            supports_multiple_asks = self.CONFIG['supports_multiple_asks'],
            requires_batch_mode = self.CONFIG['requires_batch_mode'],
            store_paramdicts = self.CONFIG['store_paramdicts'],
        )

    @classmethod
    def from_dict(cls, d:Mapping[str, Any]): return cls(**{k.lower():v for k,v in d.items()})

    def set(self, optimizer:"Optimizer"):
        optimizer.CONFIG = self.CONFIG

    @property
    def SUPPORTS_ASK(self) -> bool: return self.CONFIG["supports_ask"]
    @SUPPORTS_ASK.setter
    def SUPPORTS_ASK(self, value:bool): self.CONFIG['supports_ask'] = value

    @property
    def SUPPORTS_MULTIPLE_ASKS(self) -> bool: return self.CONFIG["supports_multiple_asks"]
    @SUPPORTS_MULTIPLE_ASKS.setter
    def SUPPORTS_MULTIPLE_ASKS(self, value:bool): self.CONFIG['supports_multiple_asks'] = value

    @property
    def REQUIRES_BATCH_MODE(self) -> bool: return self.CONFIG["requires_batch_mode"]
    @REQUIRES_BATCH_MODE.setter
    def REQUIRES_BATCH_MODE(self, value:bool): self.CONFIG['requires_batch_mode'] = value

    @property
    def STORE_PARAMDICTS(self) -> Literal['none', 'best', 'all']: return self.CONFIG["store_paramdicts"]
    @STORE_PARAMDICTS.setter
    def STORE_PARAMDICTS(self, value: Literal['none', 'best', 'all']): self.CONFIG['store_paramdicts'] = value



