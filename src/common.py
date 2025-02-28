from os.path import splitext, split

def get_filename(path: str):
    _, tail = split(path)
    name, _ = splitext(tail)
    return name