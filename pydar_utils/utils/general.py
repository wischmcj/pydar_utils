import numpy as np

def list_if(x):
    if isinstance(x,list):
        return x
    if isinstance(x,tuple):
        return list(x)
    else:
        return [x]


def poprow(my_array, pr):
    """Row popping in numpy arrays
    Input: my_array - NumPy array, pr: row index to pop out
    Output: [new_array,popped_row]"""
    i = pr
    pop = my_array[i]
    new_array = np.vstack((my_array[:i], my_array[i + 1 :]))
    return new_array, pop