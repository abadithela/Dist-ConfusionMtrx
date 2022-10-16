import pdb
import construct_MP3 as cmp
import numpy as np

if __name__ == '__main__':
    cm_fn = "conf_matrix.p"
    mini_cmp = cmp.read_confusion_matrix(cm_fn)
    n = len(mini_cmp[0][0]) # number of classes
    conf_matrix = np.zeros((n,n))
    for k, v in mini_cmp.items():
        conf_matrix += v
    print(conf_matrix)
    pdb.set_trace()
