import pickle
import os
import time

def save_obj_file(filename: str, obj):
    if os.path.dirname(filename) != '' and not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
        
def load_obj_file(filename: str):
    with open(filename, 'rb') as inp:
        obj = pickle.load(inp)
    return obj

def thread_test_func(arg1, arg2):
    print(f"First argument: {arg1}")
    time.sleep(5)
    print(f"Second argument: {arg2}")
    time.sleep(5)
    print("Function completed")