import json
import os
import sys
import numpy as np


def listRecursive(d, key):
    for k, v in d.items():
        if isinstance(v, dict):
            for found in listRecursive(v, key):
                yield found
        if k == key:
            yield v
def dp_pca_ag ( data, epsilon=1.0, delta=0.1 ):
    '''
    This function provides a differentially-private estimate using Analyze Gauss method
    of the second moment matrix of the data

    Input:

      data = data matrix, samples are in columns
      epsilon, delta = privacy parameters
      hat_A = (\epsilon, \delta)-differentially-private estimate of A = data*data'

    Example:

      >>> import numpy as np
      >>> data = np.random.rand(10)
      >>> hat_A = dp_pca_ag ( data, 1.0, 0.1 )
      [[ 1.54704321  2.58597112  1.05587101  0.97735922  0.03357301]
       [ 2.58597112  4.86708836  1.90975259  1.41030773  0.06620355]
       [ 1.05587101  1.90975259  1.45824498 -0.12231379 -0.83844168]
       [ 0.97735922  1.41030773 -0.12231379  1.47130207  0.91925544]
       [ 0.03357301  0.06620355 -0.83844168  0.91925544  1.06881321]]

    '''

    import numpy as np

    if any( np.diag( np.dot( data.transpose(), data ) ) ) > 1:
        print('ERROR: Each column in the data matrix should have 2-norm bounded in [0,1].')
        return
    elif epsilon < 0.0:
        print('ERROR: Epsilon should be positive.')
        return
    elif delta < 0.0 or delta > 1.0:
        print('ERROR: Delta should be bounded in [0,1].')
        return
    else:
        m, N = data.shape
        A = (1.0 / N) * np.dot( data, data.transpose() )
        D = ( 1.0 / (N * epsilon) ) * np.sqrt( 2.0 * np.log( 1.25 / delta ) )
        temp = np.random.normal( 0, D, (m, m))
        temp2 = np.triu( temp )
        temp3 = temp2.transpose()
        temp4 = np.tril(temp3, -1)
        E = temp2 + temp4
        hat_A = A + E
        return hat_A

def local_1(args):

    input_list = args["input"]
    myFile = input_list["samples"]
    
    # read local data
    filename = os.path.join(args["state"]["baseDirectory"], myFile)
    tmp = np.load(filename)
    Xs = tmp['arr_0']
    K = tmp['arr_3']
    cov = tmp['arr_2']
    R = 2 * K
    epsilon = 1e-2
    delta = 0.01
    
    # noisy cov matrix for privacy
    Cs = dp_pca_ag ( Xs, epsilon, delta )    
    
    # compute SVD and the partial square root
    U, S, V = np.linalg.svd(Cs)
    U = U[:, :R]
    S = S[:R]
    tmp = np.diag(np.sqrt(S))
    P = np.dot(U, tmp) 
    
    # dump outputs
    computation_output = {
        "output": {
            "psr": P.tolist(),
            "cov": cov.tolist(),
            "computation_phase": 'local_1'
        }
    }

    return json.dumps(computation_output)


if __name__ == '__main__':

#    parsed_args = json.loads(sys.argv[1])
    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(listRecursive(parsed_args, 'computation_phase'))

    if not phase_key:
        computation_output = local_1(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Local")
