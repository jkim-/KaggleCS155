#! /nfs/raid13/babar/software/anaconda/bin/python

import os
import numpy as np
import tempfile
import subprocess
import itertools
import getpass

param_list = [ 
    [ 10, 50 , 100 ], 
    [ 11, 22] 
]

for i, pars in enumerate(itertools.product(*param_list)):
    arguments = []

    # n_estimator
    arguments.append(str(pars[0]))

    # max_features
    arguments.append(str(pars[1]))

    # model_fname
    arguments.append(
        '/nfs/raid13/babar/dchao/KaggleCS155/models/test1/{0}.pkl'.format(i)
    )
#    strpars = [ str(x) for x in pars ]
#    strpars.append(str(model_num))

    submit_file = tempfile.NamedTemporaryFile(mode='w+t', suffix='.submit')
    submit_file.tempdir = '.'
    try:
        print 'Creating temporary submission file...'
        submit_file.writelines(['universe = vanilla\n',
                                'executable = test1.py\n',
                                'getenv = True\n',
                                #'arguments = '+'"'+' '.join(strpars)+'"\n',
                                'arguments = {0}\n'.format(' '.join(arguments)),
                                'output = logs/test1_'+str(i)+'.out\n',
                                'error = logs/test1_'+str(i)+'.error\n',
                                'log = logs/test1_'+str(i)+'.log\n',
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
