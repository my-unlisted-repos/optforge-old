from enum import StrEnum
class GROUPS(StrEnum):
    MAIN = 'main'
    EXTRA = 'extra'
    NO_FORCE_STOP = 'noforce'
    FREEZES = 'freezes'
    BROKEN = 'broken'
    DEPRECATED = 'deprecated'
    UNUSED = 'unused'
    DUPLICATE = 'duplicate'