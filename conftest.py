# conftest.py — pytest root configuration
# Adds the project root to sys.path so all imports work from any directory.

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
