#! /nfs/raid13/babar/software/anaconda/bin/python

import sys
model_path = '/nfs/raid13/babar/dchao/KaggleCS155'
sys.path.append(model_path)

import numpy as np
import tempfile
import subprocess
import itertools
import getpass

param_list = [
    [float(10**x) for x in range(-7, 4)], # C
    ['poly'], # kernel
    [2, 3, 5], # degree
    [200, 'None'], # n_components (for pca)
    ['True'] # whiten (for pca)
]

for i, pars in enumerate(itertools.product(*param_list)):
    arguments = []

    # C
    arguments.append(str(pars[0]))

    # kernel
    arguments.append(str(pars[1]))

    # degree 
    arguments.append(str(pars[2]))

    # n_components
    arguments.append(str(pars[3]))

    # whiten
    arguments.append(str(pars[4]))

    # model_fname
    arguments.append(
        '/nfs/raid13/babar/dchao/KaggleCS155/models/svm_poly/{0}.pkl'.format(i)
    )

    submit_file = tempfile.NamedTemporaryFile(mode='w+t', suffix='.submit')
    submit_file.tempdir = '.'
    try:
        print 'Creating temporary submission file...'
        submit_file.writelines(['universe = vanilla\n',
                                'executable = svm_poly.py\n',
                                'getenv = True\n',
                                'arguments = {0}\n'.format(' '.join(arguments)),
                                'output = '+model_path+'/models/logs/svm_poly_'+str(i)+'.out\n',
                                'error = '+model_path+'/models/logs/svm_poly_'+str(i)+'.error\n',
                                'log = '+model_path+'/models/logs/svm_poly_'+str(i)+'.log\n',
                                'accounting_group = group_babar\n',
                                'accounting_group_user = {0}\n'.format(getpass.getuser()),
                                'queue'])
        submit_file.seek(0)
        #for line in submit_file:
        #    print line.rstrip()
        #print
        subprocess.check_call(['condor_submit', submit_file.name])
    finally:
        print 'Job %d submitted!' % (i + 1)
        submit_file.close()
