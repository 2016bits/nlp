import os

def iter_files(path):
    """Walk through all files located under a root path."""
    if os.path.isfile(path):
        yield path
    elif os.path.isdir(path):
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                yield os.path.join(dirpath, f)
    else:
        raise RuntimeError('Path %s is invalid' % path)

data_path = "./data/Wikipedia/wiki-pages/"
files = [f for f in iter_files(data_path)]

for f in files:
    wiki_page = f.split('/')[-1]
    print(wiki_page)
