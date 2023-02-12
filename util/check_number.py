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
        if x is None:
            return False
        if not math.isfinite(x):
            return False
        else:
            return True

    if isinstance(x, torch.Tensor) or isinstance(x, numpy.ndarray) or isinstance(x, list):
        for i in x:
            if i is None:
                print(f"is_variable_valid gets None input as {x}.")
                return False
            if isinstance(i, str):
                print(f"is_variable_valid gets String as {x}.")
                return False
            if not math.isfinite(i):
                return False
        return True
    else:
        print(f"is_variable_valid gets not qualified input as {x}.")
        return False

    # if isinstance(x, torch.Tensor):
    #     for i in x:
    #         if i is None:
    #             print(f"is_variable_valid gets None input as {x}.")
    #             return False
    #         if isinstance(i, str):
    #             print(f"is_variable_valid gets String as {x}.")
    #             return False
    #         if not torch.isfinite(i):
    #             return False
    #     return True
    # elif isinstance(x, numpy.ndarray):
    #     for i in x:
    #         if i is None:
    #             print(f"is_variable_valid gets None input as {x}.")
    #             return False
    #         if isinstance(i, str):
    #             print(f"is_variable_valid gets String as {x}.")
    #             return False
    #         if not numpy.isfinite(i):
    #             return False
    #     return True
    # elif isinstance(x, list):
    #     for i in x:
    #         if i is None:
    #             print(f"is_variable_valid gets None input as {x}.")
    #             return False
    #         if isinstance(i, str):
    #             print(f"is_variable_valid gets String as {x}.")
    #             return False
    #         if not math.isfinite(i):
    #             return False
    #     return True
    # else:
    #     print(f"is_variable_valid gets not qualified input as {x}.")
    #     return False

if __name__=='__main__':
    a = [None, 1, 2]
    np_arr = numpy.array(a, dtype=float)
    tensor_arr = torch.from_numpy(np_arr)
    assert is_variable_valid(a)==False, "Checking function wrong."
    assert is_variable_valid(np_arr)==False, "Checking function wrong."
    assert is_variable_valid(tensor_arr)==False, "Checking function wrong."

    a = 1.0
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

    a = "qwer"
    assert is_variable_valid(a)==False, "Checking function wrong."

    a = ["qwer"]
    assert is_variable_valid(a)==False, "Checking function wrong."

    a = ["qwer", 1]
    assert is_variable_valid(a)==False, "Checking function wrong."