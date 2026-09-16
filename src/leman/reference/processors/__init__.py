# processors/__init__.py
from importlib import import_module
from pathlib import Path

for _path in Path(__file__).parent.glob("*.py"):
    if _path.stem != "__init__":
        from importlib import import_module
        _mod = import_module(f".{_path.stem}", package=__name__)
        globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("_")})

# Explicit imports for static analysis (Pylance/VSCode)
from .Tagarelli2023 import *
from .Vaquero2026 import *
from .Dijkstra2025 import *
from .Alexeev2019 import *
from .Lin2024 import *        
from .Louca2023 import *