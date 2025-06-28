import os
def get_file(filename):
    abs = os.path.abspath(__file__)
    dirname,_ = os.path.split(abs)
    return os.path.join(dirname, filename)