
def get_precision(X:list, Y:list):
    _intersection = set(X).intersection(Y)
    return len(_intersection) / len(Y)

def get_recall(X:list, Y:list):
    _intersection = set(X).intersection(Y)
    return len(_intersection) / len(X)

