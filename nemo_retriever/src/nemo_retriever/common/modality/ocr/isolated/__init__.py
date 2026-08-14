# SPDX-License-Identifier: Apache-2.0

"""Opt-in OCR pipelines isolated from the default graph.

The default ingestor does not import these implementations.  The service
worker imports the graph adapter only after an explicit Option 3/4/5 selector.
"""

from nemo_retriever.common.modality.ocr.isolated.contracts import (
    OCRPage,
    OCRPageOutput,
    OCRUnit,
)
from nemo_retriever.common.modality.ocr.isolated.option3 import (
    Option3Config,
    Option3Pipeline,
)
from nemo_retriever.common.modality.ocr.isolated.option4 import (
    Option4Config,
    Option4Pipeline,
)
from nemo_retriever.common.modality.ocr.isolated.option5 import (
    Option5Config,
    Option5Pipeline,
)
from nemo_retriever.common.modality.ocr.isolated.option7 import (
    Option7Config,
    Option7Pipeline,
)

__all__ = [
    "OCRPage",
    "OCRPageOutput",
    "OCRUnit",
    "Option3Config",
    "Option3Pipeline",
    "Option4Config",
    "Option4Pipeline",
    "Option5Config",
    "Option5Pipeline",
    "Option7Config",
    "Option7Pipeline",
]
