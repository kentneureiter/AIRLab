import os
import sys
print("Files on board:")
print(os.listdir())
print("\nSystem info:")
print(sys.version)
try:
    import ulab
    print("ulab: installed", ulab.__version__)
except ImportError:
    print("ulab: NOT installed")
try:
    from ulab import numpy as np
    print("numpy: available")
except ImportError:
    print("numpy: NOT available")
