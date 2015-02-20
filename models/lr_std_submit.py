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
    ['l1', 'l2'], # penalty
    [float(10**x) for x in range(-5,3)], # C
    ['True', 'False'], # fit_intercept
    ['200', 'None'] # n_components for PCA
]

for i, pars in enumerate(itertools.product(*param_list)):
    arguments = []

    # penalty 
    arguments.append(str(pars[0]))

    # C
    arguments.append(str(pars[1]))

    # fit_intercept
    arguments.append(str(pars[2]))

    # n_components
    arguments.append(str(pars[3]))

    # model_fname
    arguments.append(
        '/nfs/raid13/babar/dchao/KaggleCS155/models/lr_std/{0}.pkl'.format(i)
    )

    submit_file = tempfile.NamedTemporaryFile(mode='w+t', suffix='.submit')
    submit_file.tempdir = '.'
    try:
        print 'Creating temporary submission file...'
        submit_file.writelines(['universe = vanilla\n',
                                'executable = lr_std.py\n',
                                'getenv = True\n',
                                'arguments = {0}\n'.format(' '.join(arguments)),
                                'output = '+model_path+'/models/logs/lr_std_'+str(i)+'.out\n',
                                'error = '+model_path+'/models/logs/lr_std_'+str(i)+'.error\n',
                                'log = '+model_path+'/models/logs/lr_std_'+str(i)+'.log\n',
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
