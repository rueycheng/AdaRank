import numpy as np
import re
import sys


def group_counts(arr):
    d = np.ones(arr.size, dtype=int)
    d[1:] = (arr[:-1] != arr[1:]).astype(int)
    return np.diff(np.where(np.append(d, 1))[0])


def group_offsets(arr):
    """Return a sequence of start/end offsets for the value subgroups in the input"""
    d = np.ones(arr.size, dtype=int)
    d[1:] = (arr[:-1] != arr[1:]).astype(int)
    idx = np.where(np.append(d, 1))[0]
    return zip(idx, idx[1:])


def load_docno(fname, letor=False):
    """Load docnos from the input in the SVMLight format"""
    if letor:
        docno_pattern = re.compile(r'#\s*docid\s*=\s*(\S+)')
    else:
        docno_pattern = re.compile(r'#\s*(\S+)')

    docno = []
    for line in open(fname):
        if line.startswith('#'):
            continue
        m = re.search(docno_pattern, line)
        if m is not None:
            docno.append(m.group(1))
    return np.array(docno)


def print_trec_run(qid, docno, pred, run_id='exp', output=None):
    """Print TREC-format run to output"""
    if output is None:
        output = sys.stdout
    for a, b in group_offsets(qid):
        idx = np.argsort(-pred[a:b]) + a  # note the minus and plus a
        for rank, i in enumerate(idx, 1):
            output.write('{qid} Q0 {docno} {rank} {sim} {run_id}\n'.
                         format(qid=qid[i], docno=docno[i], rank=rank, sim=pred[i], run_id=run_id))
