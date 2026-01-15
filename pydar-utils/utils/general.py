
def list_if(x):
    if isinstance(x,list):
        return x
    if isinstance(x,tuple):
        return list(x)
    else:
        return [x]
