"""
vabert — visual acuity extraction from UK ophthalmology clinic letters.

    from vabert import VABERTPipeline
    result = VABERTPipeline().run(text)   # {'status': ..., 'readings': [...]}

Command line: `vabert --help` (or `python -m vabert --help`).
"""

from .pipeline import VABERTPipeline
from .inference import FIELDS

__version__ = '1.0.0'
__all__ = ['VABERTPipeline', 'FIELDS', '__version__']
