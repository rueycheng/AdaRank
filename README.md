# AdaRank

This is a python implementation of the AdaRank algorithm (Xu and Li, 2007) with early stopping.

The structure of the code follows closely to the scikit-learn style, but still there are some
differences in the estimator/metrics API (e.g. `fit()` method takes three arguments `X`, `y`,
and `qid` rather than just two).

Four ranking metrics are implemented: P@k, AP, DCG@k, and nDCG@k
(in both `trec_eval` and Burges et al. versions). 

## Dependencies

* `numpy`
* `scikit-learn`

## Usage

The following code will run AdaRank for 100 iterations optimizing for NDCG@10.  When no
improvement is made within the previous 10 iterations, the algorithm will stop.

```
from adarank import AdaRank
from metrics import NDCGScorer

scorer = NDCGScorer(k=10)
model = AdaRank(max_iter=100, estop=10, scorer=scorer).fit(X, y, qid)
pred = model.predict(X_test, qid_test)
print scorer(y_test, pred, qid_test).mean()
```

See [test.py](test.py) for more advanced examples.

## References

Burges et al. Learning to rank using gradient descent. In 
_Proceedings of ICML '05_, pages 89&ndash;96. ACM, 2005.

Xu and Li. AdaRank: a boosting algorithm for information retrieval. In
_Proceedings of SIGIR '07_, pages 391&ndash;398. ACM, 2007.

