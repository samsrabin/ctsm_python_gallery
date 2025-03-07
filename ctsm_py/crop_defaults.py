"""
Some useful crop default variables
"""

from types import MappingProxyType

# MappingProxyType makes this dict immutable
DEFAULT_VAR_DICT = MappingProxyType(
    {
        "hui_var": "HUI",
        "huifrac_var": "HUIFRAC",
        "gddharv_var": "GDDHARV",
        "gslen_var": "GSLEN",
    }
)
