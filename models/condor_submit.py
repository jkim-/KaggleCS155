import os
import numpy as np
import tempfile
import subprocess
import itertools

param_list = [[300, 500], # n_estimators
              [44],#[int(2**x * np.sqrt(200)) for x in range(-1, 4)], # max_features
              [None],#[None, 1, 2], # max_depth
              [1, 2], # min_samples_leaf
              [False]] # bootstrap

model_num = 0

for pars in itertools.product(*param_list):
    strpars = [str(x) for x in pars]
    strpars.append(str(model_num))
    submit_file = tempfile.NamedTemporaryFile(mode='w+t', suffix='.submit')
    submit_file.tempdir = '.'
    try:
        print 'Creating temporary submission file...'
        submit_file.writelines(['universe = vanilla\n',
                                'executable = rf1.py\n',
                                'arguments = '+'"'+' '.join(strpars)+'"\n',
                                'output = batch/rf1_'+str(model_num)+'.out\n',
                                'error = batch/rf1_'+str(model_num)+'.error\n',
                                'log = batch/rf1_'+str(model_num)+'.log\n',
                                'accounting_group = group_babar\n',
                                'accounting_group_user = jkim\n',
                                #'getenv = True\n',
                                'when_to_transfer_output = ON_EXIT\n',
                                'transfer_input_files = models, pyutils, data\n',
                                'queue'])
        submit_file.seek(0)
        #for line in submit_file:
        #    print line.rstrip()
        subprocess.check_call(['condor_submit', submit_file.name])
    finally:
        print 'Job %d submitted!' % model_num
        submit_file.close()
        model_num += 1
