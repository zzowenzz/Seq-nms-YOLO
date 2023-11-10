import numpy as np
import re

def atoi(text):
    """
    A helper function to convert a string to integer if it is a digit
    """
    return int(text) if text.isdigit() else text

def natural_keys(text):
    """
    A helper function to generate a key for sorting strings containing numerical values in human order
    """
    return [atoi(c) for c in re.split(r'(\d+)', text)]