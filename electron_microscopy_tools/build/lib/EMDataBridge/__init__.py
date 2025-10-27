#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EMDataBridge: A bridge between electron microscopy data formats

This package provides tools for standardizing, translating, and discovering
electron microscopy data formats and their metadata.
"""

__version__ = "0.0.1"
__author__ = "Roberto dos Reis"
__email__ = "robertomsreis@gmail.com"
__url__ = "https://github.com/rmsreis/EMDataBridge"

# Import main classes for easier access
from .standardizer import EMDataStandardizer
from .translator import EMFormatTranslator
from .discovery import EMDataDiscovery
