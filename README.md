# spark-node2vec

Simplified implementation of [Node2vec](https://snap.stanford.edu/node2vec/) in Spark without random walks. Essentially now equivalent to Word2Vec on graph.

`spark-submit --class Node2Vec build.jar train_epin.csv test_epin.csv`
