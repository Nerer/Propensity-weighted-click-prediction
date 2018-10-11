import os

tot = 0
with open('history.txt', 'w') as f:
    f.write('')

    for algo in ['mlp', 'pmlpb1', 'pmlpb2h', 'pmlpb2']:
        for clipping in [1, 1.5, 2, 3, 4, 8, 16, 32]:
            for wd in [1e-5, 3e-5, 5e-5, 1e-4, 3e-4, 5e-4, 1e-6, 3e-6, 5e-6]:
                print('python main.py -alg %s -pred -com -wd %f -wclip %f -d'%(algo, wd, clipping))
                os.system('python main.py -alg %s -pred -com -wd %f -wclip %f -d'%(algo, wd, clipping))