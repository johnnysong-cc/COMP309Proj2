# -*- coding: utf-8 -*-
"""
Model Pickler / Depickler Module

Section 2.5.2: Using the pickle module, arrange for Serialization 
& Deserialization of your model.
"""
import pickle as pkl
class Pickler():
    
    # Serialize / Pickle the model
    @staticmethod
    def pickle(model):
        return pkl.dumps(model, protocol=pkl.HIGHEST_PROTOCOL, 
                            fix_imports=True, buffer_callback=None)
    
    # Deserialize / Depickle the model
    @staticmethod
    def depickle(pickle):
        return pkl.loads(pickle, fix_imports=True,
                            errors='strict', buffers=None)
    
    # Serializes and saves the model to the given path
    @staticmethod
    def jar(model, path):
        with open(path, 'w') as file:
            return pkl.dump(model, file, protocol=None, 
                    fix_imports=True, buffer_callback=None)
        
    # Deserializes a pickle from the given path
    @staticmethod
    def unjar(path):
        with open(path, 'r') as file:
            return pkl.load(file, fix_imports=True,
                               errors='strict', buffers=None)