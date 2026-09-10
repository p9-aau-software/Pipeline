"""Model implementations.

Every module in this package is imported on startup so its ``@MODELS.register``
decorator runs. A module that fails to import (a missing optional dependency, say) is
skipped and the reason is reported when someone asks for a model it would have provided.
"""

from recpipe.registry import autoload

autoload(__name__, skip=("base", "external"))
