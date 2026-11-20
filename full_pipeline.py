#!/usr/bin/env python3
"""
full_pipeline.py - Entrypoint alias for full_pipline.py
Ensures compatibility with both 'python full_pipeline.py' and 'python full_pipline.py'.
"""

import sys
from full_pipline import main

if __name__ == "__main__":
    main()
