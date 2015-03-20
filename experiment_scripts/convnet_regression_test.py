# A simple script to check whether two implementations are equivalent.

import sys, os
import numpy as np
import numpy.random as npr

from deepmolecule import get_data_file, load_data
from deepmolecule import build_universal_net, build_universal_net_ag

def main():

    os.environ['DATA_DIR'] = "/Users/dkd/repos/DeepMoleculesData/data"

    # Parameters for convolutional net.
    conv_arch_params = {'num_hidden_features' : [5, 5, 12],
                        'permutations' : True}

    task_params = {'N_train'     : 200,
                   'target_name' : 'Log Rate',
                   'data_file'   : get_data_file('2014-11-03-all-tddft/processed.csv')}

    print "Conv net architecture params: ", conv_arch_params
    print "Task params", task_params

    print "\nLoading data..."
    (traindata, ) = load_data(task_params['data_file'], (task_params['N_train'],))
    train_inputs, train_targets = traindata['smiles'], traindata[task_params['target_name']]

    loss_fun1, grad_fun1, pred_fun, hiddens_fun1, parser1 = build_universal_net_ag(**conv_arch_params)
    npr.seed(0)
    weights = npr.randn(parser1.N)
    print "Outputs: ", pred_fun(weights, train_inputs)


if __name__ == '__main__':
    sys.exit(main())

"""Should produce:

Outputs:  [ -23.1073787   -27.73925982  -15.95001241    9.43008428  -29.4289622
   29.23967409  -29.51986456   15.83570718  -29.97904165   14.70702161
   44.47499715   -8.09419247    9.62896985   -6.5415892     5.18802393
   22.10104161    4.85848102  -33.75224996    1.08850371   -0.56449294
  -15.64185267   -9.51712084  -18.29910983   -3.30174189   12.70019039
 -129.19096471  -21.0188696    -9.90969119  -18.33349506   -2.46542496
    0.55234527    1.12186231  -10.3778619     4.84902574    5.70566639
   -0.29364022    5.77972146   -9.01712276  -15.69371002  -24.90802273
    6.57434727  -60.71402352   -8.47753976   41.57410471   42.22982291
  -29.17098116   12.73027469    6.37610417   21.39441739   20.70475793
   21.7680409   -37.469596     19.52630258   33.66910569   12.9132224
  -12.55593975   13.52367924    4.8725331     0.43631967   36.88916075
   10.63322403  -15.83105321   16.7652141    10.15119264   -9.88570643
   20.15431872   -3.35821751   -8.31029025    5.6980197    20.0293763
   -3.98620912  -34.36810372   -8.16875897   14.86505642   26.02384855
  -17.06206392   30.19187545    9.82550221  -15.52242527  -27.53569273
  -11.2185141    -7.19738692  -11.09798332  -45.46507058  -28.07172329
    7.66660424    2.54672538  -33.2901825     6.18817795   -3.27790203
  -22.8647052    -5.75887243  -30.15674239  -23.74309575   15.68922423
   -1.83559469   30.16702574    6.59757284  -12.16252037    1.05517309
  -10.94562956   34.85211784   30.22516731   37.76937683    3.45716739
   10.69134033  -68.90545998   -1.02529443   28.86336192    2.19527143
  -77.62198765  -37.18166517    5.20790004  -36.08119784    5.4653965
    0.89850127 -135.02448531  -23.23729228  -17.8383304    17.90941515
   21.09152033   11.71951885   27.95477961   28.740525     -6.54982807
  -61.54270026   19.81673885   10.65547115   10.69505113   -8.85713267
   12.04262164   -2.87666474    6.33559158  -18.6080246   -41.38143454
   24.50070885  -39.37905316  -41.55520061  -70.80172955   23.95734037
  -74.26536042  -17.8482132    11.92937683   10.92519035  -51.55142298
   -8.51607207   13.46729325    6.41158015  -62.66470117  -12.11727267
  -25.87838267   -2.30007509  -18.40730176  -24.2320775    -6.79687568
   14.33679815    9.47397173   28.40145472   44.52954915   -4.14467862
   -5.13123088  -21.42651303  -18.92680626  -30.80652011   25.9272126
  -14.32856543 -131.71209526    0.87032221  -17.06084234   -0.27909381
    8.87556191  -34.66602362  -27.68790005  -25.04533433  -14.14179003
   14.90774715  -24.01799319   23.29792575  -26.32166311   14.20612625
  -17.66185351   11.47717634    2.34613583   16.16780483    9.84241793
  -24.31693057   -9.12045296  -78.93800274  -20.80617267  -10.83557262
  -10.84057835   -1.49921861  -13.71855903   -0.13744958  -12.15664157
    9.93213847   -2.41912645  -20.59484119  -16.9646424    34.66709838]
"""
