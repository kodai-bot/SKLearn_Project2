import glob
import os
import pandas as pd

class DataCombiner:

    def __init__(self, filepath):
      self.filepath = filepath

    def combine_csv(self):
      # Get a list of all csv files in the directory
      files = glob.glob(self.filepath)

      # Sort the list of files alphabetically
      files = sorted(files)
      return files

# add json file to the combined dataframe


# Visulaisation abd insights

# Feature engineering


