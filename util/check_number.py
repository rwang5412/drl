import torch
import numpy
import math

def is_variable_valid(x):
    """Check if a given number has NaN, Inf, or None. 
    Datatype of x in list, tensor, ndarray, floating number
    Returns: True if this number of array has the above values.
    """
    try: 
        iter(x)
    except:
        if x is None or not math.isfinite(x):
            return False
        else:
            return True
    else:
        if any(i is None for i in x):
            return False
    
    if isinstance(x, torch.Tensor):
        for i in x:
            if not torch.isfinite(i):
                return False
        return True
    elif isinstance(x, numpy.ndarray):
        for i in x:
            if not numpy.isfinite(i):
                return False
        return True
    elif isinstance(x, list):
        for i in x:
            if not math.isfinite(i):
                return False
        return True
    else:
        print(x, "new")
    
if __name__=='__main__':
    a = [None, 1, 2]
    np_arr = numpy.array(a, dtype=float)
    tensor_arr = torch.from_numpy(np_arr)
    assert is_variable_valid(a)==False, "Checking function wrong."
    assert is_variable_valid(np_arr)==False, "Checking function wrong."
    assert is_variable_valid(tensor_arr)==False, "Checking function wrong."
    
    a = [1]
    np_arr = numpy.array(a, dtype=float)
    tensor_arr = torch.from_numpy(np_arr)
    assert is_variable_valid(a)==True, "Checking function wrong."
    assert is_variable_valid(np_arr)==True, "Checking function wrong."
    assert is_variable_valid(tensor_arr)==True, "Checking function wrong."
    
    a = [1, float('Inf')]
    np_arr = numpy.array(a, dtype=float)
    tensor_arr = torch.from_numpy(np_arr)
    assert is_variable_valid(a)==False, "Checking function wrong."
    assert is_variable_valid(np_arr)==False, "Checking function wrong."
    assert is_variable_valid(tensor_arr)==False, "Checking function wrong."
