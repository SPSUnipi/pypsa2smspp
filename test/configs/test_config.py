# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:10:34 2024

@author: aless
"""

from configparser import ConfigParser
import os

DIR = os.path.dirname(os.path.abspath(__file__))

# Class for the initial configuration of the project
class TestConfig():
    __test__ = False  # disable test

    def __init__(self, fp="application.ini"):
        fp = os.path.join(DIR, fp)
        self.parser = ConfigParser()
        self.parser.read(fp)

        self.parser["INPUT"]["input_data_path"] = os.path.abspath(
            os.path.join(DIR, self.parser["INPUT"]["input_data_path"])
        )
        
        self.set_attributes()

#%% Method to dinamically read the parameters in the config file
    def set_attributes(self):
        for section in self.parser.sections():
            for key in self.parser[section]:
                self.attribute_definition(key, self.parse_value(self.parser[section][key]))

#%% Method to understand the type of the parsed value
    def parse_value(self, value):
        try:
            # Try to evaluate the value
            evaluated_value = eval(value)
            # If the evaluated value is not a string, return it
            if not isinstance(evaluated_value, str):
                return evaluated_value
        except:
            # If evaluation fails, return the original string
            pass
        return value

#%% Method to set the attribute of config
    def attribute_definition(self, key, value):
        setattr(self, key, value)