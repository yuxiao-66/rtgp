import pickle
import os 

def save_pickle(x, filename):
    # mkdir(filename)
    with open(filename, 'wb') as file:
        pickle.dump(x, file)
    print('save',filename)


def load_pickle(filename):
    with open(filename, 'rb') as file:
        x = pickle.load(file)
    print(f'Pickle loaded from {filename}')
    return x

def write_text(obj, filename, append=False):
    mkdir(filename)
    mode = 'a' if append else 'w'
    with open(filename, mode) as file:
        print(obj, file=file)
    print(filename)

def mkdir(path):
    dirname = os.path.dirname(path)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)