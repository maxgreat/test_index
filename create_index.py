import faiss
import numpy as np
import argparse
import re
from tqdm import trange
import time


def choose_train_size(index_key):

    # some training vectors for PQ and the PCA
    n_train = 65000

    if "IVF" in index_key:
        matches = re.findall("IVF([0-9]+)", index_key)
        ncentroids = int(matches[0])
        n_train = max(n_train, 100 * ncentroids)
    elif "IMI" in index_key:
        matches = re.findall("IMI2x([0-9]+)", index_key)
        nbit = int(matches[0])
        n_train = max(n_train, 256 * (1 << nbit))
    return 65000
    return n_train


parser = argparse.ArgumentParser()
parser.add_argument("--nb_features", default=100000, type=int)
parser.add_argument("--index_factory", default="OPQ8_32,IVF262144,PQ8", type=str)
parser.add_argument("--n_train", default=-1, type=int)
parser.add_argument("--output_dir", required=True, type=str)
parser.add_argument("--index_version", required=True, type=str)
parser.add_argument("--index_size", default=-1, type=int)
parser.add_argument("--row_size", default=2048, type=int)

args = parser.parse_args()

index_factory = args.index_factory
output_dir = args.output_dir
index_file = index_factory.replace(",", "_") + args.index_version + ".index"


# index = faiss.index_factory(args.row_size, index_factory, faiss.METRIC_INNER_PRODUCT)
index = faiss.index_factory(args.row_size, index_factory)
n_train = choose_train_size(index_factory)
n_train = min(n_train, args.nb_features)
xt = np.random.random((n_train, args.row_size)).astype("float32").copy()
print("Using " + str(len(xt)) + " vectors to train")
index.verbose = True
t0 = time.time()
index.train(xt)
print("train done in %.3f s" % (time.time() - t0))
# faiss.write_index(index, output_dir+'/trained_'+index_file)

index_with_id = faiss.IndexIDMap(index)


for i in range(args.nb_features // n_train):
    a = np.random.random((n_train, args.row_size)).astype("float32")
    b = np.arange(start=i, stop=i + n_train).astype("long")
    index_with_id.add_with_ids(a, b)

# Extract a groundtruth and train the index using this groundtruth.
# Use Flat index for groundtruth
print(index_with_id.ntotal)
faiss.write_index(index_with_id, output_dir + "/" + index_file)

# if 'OPQ' in index_factory:
#     ps = faiss.ParameterSpace()
#     ps.initialize(index_with_id)
#     # setup the Criterion object: optimize for 1-R@1
#    crit = faiss.OneRecallAtRCriterion(nq, 1)
#    # by default, the criterion will request only 1 NN
#    crit.nnn = 100
#    crit.set_groundtruth(None, gt.astype('int64'))

#    # then we let Faiss find the optimal parameters by itself
#    print "exploring operating points"

#    t0 = time.time()
#    op = ps.explore(index, xq, crit)
#    print "Done in %.3f s, available OPs:" % (time.time() - t0)

#    # opv is a C++ vector, so it cannot be accessed like a Python array
#    opv = op.optimal_pts
#    print "%-40s  1-R@1     time" % "Parameters"
#    for i in range(opv.size()):
#        opt = opv.at(i)
#        print "%-40s  %.4f  %7.3f" % (opt.key, opt.perf, opt.t)

