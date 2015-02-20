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
    ['deviance', 'ls'], # loss
    [float(10**x) for x in range(-2,1)], # learning_rate
    [64, 128, 180, 300, 400, 600, 700, 800], # n_estimators
    [1, 3, 5], # max_depth
    [0.5] # subsample
]

for i, pars in enumerate(itertools.product(*param_list)):
    arguments = []

    # loss
    arguments.append(str(pars[0]))

    # learning_rate
    arguments.append(str(pars[1]))

    # n_estimators 
    arguments.append(str(pars[2]))

    # max_depth
    arguments.append(str(pars[3]))
    
    # subsample
    arguments.append(str(pars[4]))
    
    # model_fname
    arguments.append(
        '/nfs/raid13/babar/dchao/KaggleCS155/models/gtb_std/{0}.pkl'.format(i)
    )

    submit_file = tempfile.NamedTemporaryFile(mode='w+t', suffix='.submit')
    submit_file.tempdir = '.'
    try:
        print 'Creating temporary submission file...'
        submit_file.writelines(['universe = vanilla\n',
                                'executable = gtb_std.py\n',
                                'getenv = True\n',
                                'arguments = {0}\n'.format(' '.join(arguments)),
                                'output = '+model_path+'/models/logs/gtb_std_'+str(i)+'.out\n',
                                'error = '+model_path+'/models/logs/gtb_std_'+str(i)+'.error\n',
                                'log = '+model_path+'/models/logs/gtb_std_'+str(i)+'.log\n',
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
