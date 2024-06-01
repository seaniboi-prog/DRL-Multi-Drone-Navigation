import pickle
import os

def save_obj_file(filename: str, obj):
    if os.path.dirname(filename) != '' and not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
        
def load_obj_file(filename: str):
    with open(filename, 'rb') as inp:
        obj = pickle.load(inp)
    return obj