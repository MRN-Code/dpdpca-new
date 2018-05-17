import json
import sys
import numpy as np


def listRecursive(d, key):
    for k, v in d.items():
        if isinstance(v, dict):
            for found in listRecursive(v, key):
                yield found
        if k == key:
            yield v


def remote_1(args):

    input_list = args["input"]
#    myval = [input_list[site]["output_val"] for site in input_list]
    sums = 0
    for site in input_list:
        Ps = np.array(input_list[site]["psr"])
        sums = sums + np.dot(Ps, Ps.T)
        
    sums = sums / len(input_list)

    u, s, v = np.linalg.svd(sums)
    K = 2
    u = u[:, :K]
    
    cov = np.array(input_list[site]["cov"])
    en = np.trace(np.dot(np.dot(u.T, cov), u))

    computation_output = {"output": {"en": en}, "success": True}
    return json.dumps(computation_output)


if __name__ == '__main__':

#    parsed_args = json.loads(sys.argv[1])
    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(listRecursive(parsed_args, 'computation_phase'))

    if "local_1" in phase_key:
        computation_output = remote_1(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Remote")
