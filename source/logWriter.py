def init(file):
    with open(file, 'w') as f:
        f.write('')

def write(text, file, verbose = False):
    with open(file, 'a') as f:
        f.write(str(text) + "\n")
    if (verbose):
        print(text)