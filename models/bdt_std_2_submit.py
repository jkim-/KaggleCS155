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
    [int(1.2**x) for x in range(25,40)], # n_estimators
    [2, 3, 4, 5], # max_depth 
    [0.7, 1.] # learning_rate
]

for i, pars in enumerate(itertools.product(*param_list)):
    arguments = []

    # n_estimators
    arguments.append(str(pars[0]))

    # max_depth
    arguments.append(str(pars[1]))

    # learning_rate
    arguments.append(str(pars[2]))

    # model_fname
    arguments.append(
        '/nfs/raid13/babar/dchao/KaggleCS155/models/bdt_std_2/{0}.pkl'.format(i)
    )

    submit_file = tempfile.NamedTemporaryFile(mode='w+t', suffix='.submit')
    submit_file.tempdir = '.'
    try:
        print 'Creating temporary submission file...'
        submit_file.writelines(['universe = vanilla\n',
                                'executable = bdt_std_2.py\n',
                                'getenv = True\n',
                                'arguments = {0}\n'.format(' '.join(arguments)),
                                'output = '+model_path+'/models/logs/bdt_std_2_'+str(i)+'.out\n',
                                'error = '+model_path+'/models/logs/bdt_std_2_'+str(i)+'.error\n',
                                'log = '+model_path+'/models/logs/bdt_std_2_'+str(i)+'.log\n',
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
