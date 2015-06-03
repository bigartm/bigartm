<p align="center">
	<img alt="BigARTM Logo" src="http://bigartm.org/img/BigARTM-logo.svg" width="250">
</p>

The state-of-the-art platform for topic modeling.

[![Build Status](https://secure.travis-ci.org/bigartm/bigartm.png)](https://travis-ci.org/bigartm/bigartm)
[![GitHub license](https://img.shields.io/badge/license-New%20BSD-blue.svg)](https://raw.github.com/bigartm/bigartm/master/LICENSE)

  - [Full Documentation](http://docs.bigartm.org/)
  - [User Mailing List](https://groups.google.com/forum/#!forum/bigartm-users)
  - [Download Releases](https://github.com/bigartm/bigartm/releases)


# What is BigARTM?

BigARTM is a tool for [topic modeling](https://www.cs.princeton.edu/~blei/papers/Blei2012.pdf) based on a novel technique called Additive Regularization of Topic Models. This technique effectively builds multi-objective models by adding the weighted sums of regularizers to the optimization criterion. BigARTM is known to combine well very different objectives, including sparsing, smoothing, topics decorrelation and many others. Such combinations of regularizers significantly improves several quality measures at once almost without any loss of the perplexity.

Here are some examples of when you could use BigARTM:

  - Extract highly-interpretable topics from text collections. [Demo]()
  - Scale up topic extraction using parallel and distributed mode. [Demo]()
  - Construct accurate text classifiers by very small training set. [Demo]()
  - Query similar documents on different languages.


Put here some defails of ARTM approach, regularizers and essential theoretical background.

### References

1. Vorontsov K., Potapenko A., Plavin A. [Additive Regularization of Topic Models for Topic Selection and Sparse Factorization.](https://s3-eu-west-1.amazonaws.com/artm/voron15slds.pdf) // Statistical Learning and Data Sciences. 2015 — pp. 193-202.
2. Vorontsov K., Frei O., Apishev M., Romov P., Dudarenko M. BigARTM: Open Source Library for Regularized Multimodal Topic Modeling of Large Collections Analysis of Images, Social Networks and Texts. 2015 [Slides](https://s3-eu-west-1.amazonaws.com/artm/voron15aist-slides.pdf)
3. Vorontsov K. V., Potapenko A. A. [Additive Regularization of Topic Models](https://s3-eu-west-1.amazonaws.com/artm/voron-potap14artm-eng.pdf) // Machine Learning Journal, Special Issue “Data Analysis and Intelligent Optimization”, Springer, 2014.

### Related Software Packages

- [David Blei's List](https://www.cs.princeton.edu/~blei/topicmodeling.html) of Open Source topic modeling software
- [MALLET](http://mallet.cs.umass.edu/topics.php): Java-based toolkit for language processing with topic modeling package
- [Gensim](https://radimrehurek.com/gensim/): Python topic modeling library
- [Vowpal Wabbit](https://github.com/JohnLangford/vowpal_wabbit) has an implementation of [Online-LDA algorithm](https://github.com/JohnLangford/vowpal_wabbit/wiki/Latent-Dirichlet-Allocation)


# How to Use

### Installing

Download [binary release](https://github.com/bigartm/bigartm/releases) or build from source using cmake:
```bash
$ mkdir build && cd build
$ cmake ..
$ make
```

### Command-line interface

**TODO** Command line interface is under construction. Check out [documentation for `cpp_client`](http://docs.bigartm.org/en/latest/ref/cpp_client.html) — the old CLI.

```bash
$ bigartm learn --format bow -d docword.kos.txt -v vocab.kos.txt
```

### Interactive Python interface

**TODO** Python API is under construction, see [documentation for the old Python interface](http://docs.bigartm.org/en/latest/tutorials/typical_python_example.html) and [examples of using BigARTM from python](src/python/examples).

```python
# A stub
import bigartm

model = bigartm.Model()
model.em_iterations(batches, n_iters=5)
model.show()
```

### Low-level API

  - [C++ Interface](http://docs.bigartm.org/en/latest/ref/cpp_interface.html)
  - [Plain C Interface](http://docs.bigartm.org/en/latest/ref/c_interface.html)


## Contributing

Refer to the [Developer's Guide](http://docs.bigartm.org/en/latest/devguide.html).

To report a bug use [issue tracker](https://github.com/bigartm/bigartm/issues). To ask a question use [our mailing list](https://groups.google.com/forum/#!forum/bigartm-users). Feel free to make [pull request](https://github.com/bigartm/bigartm/pulls).


## License

BigARTM is released under [New BSD License](https://raw.github.com/bigartm/bigartm/master/LICENSE) that allowes unlimited redistribution for any purpose (even for commercial use) as long as its copyright notices and the license’s disclaimers of warranty are maintained.
