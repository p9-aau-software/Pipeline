"""Dataset ingests: raw files of any shape -> the canonical interaction table.

Adding a dataset means adding one module here with an ``@INGESTS.register("name")``
function that returns a DataFrame with the columns in ``recpipe.data.schema``. Nothing
downstream needs to know the dataset exists.
"""

from recpipe.registry import autoload

autoload(__name__)
