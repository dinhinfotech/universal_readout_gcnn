# -*- coding: utf-8 -*-
"""
Util file includes utility functions
"""
from os import listdir
from os.path import isfile, join
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
    
def list_files_in_folder(folder_path):    
    """
    Return: A list of the file names in the folder
    """
          
    list = listdir(folder_path)
    onlyfiles = [ f for f in list  if isfile(join(folder_path,f)) ]
    return onlyfiles 

    
def load_list_from_file(file_path):
    """
    Return: A list saved in a file
    """
    
    f = open(file_path,'r')
    listlines = [line.rstrip() for line in f.readlines()]
    f.close()
    return listlines