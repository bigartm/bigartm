<p align="center">
	<img alt="BigARTM Logo" src="http://bigartm.org/img/BigARTM-logo.svg" width="250">
</p>

The state-of-the-art platform for topic modeling.

[![Build Status](https://secure.travis-ci.org/bigartm/bigartm.png)](https://travis-ci.org/bigartm/bigartm)
[![Windows Build Status](https://ci.appveyor.com/api/projects/status/i18k840shuhr2jtk/branch/master?svg=true)](https://ci.appveyor.com/project/bigartm/bigartm)
[![GitHub license](https://img.shields.io/badge/license-New%20BSD-blue.svg)](https://raw.github.com/bigartm/bigartm/master/LICENSE.txt)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.288960.svg)](https://doi.org/10.5281/zenodo.288960)

  - [Full Documentation](http://docs.bigartm.org/)
  - [User Mailing List](https://groups.google.com/forum/#!forum/bigartm-users)
  - [Download Releases](https://github.com/bigartm/bigartm/releases)
  - [User survey](http://goo.gl/forms/tr5EsPMcL2)


# What is BigARTM?

BigARTM is a powerful tool for [topic modeling](https://en.wikipedia.org/wiki/Topic_model) based on a novel technique called Additive Regularization of Topic Models. This technique effectively builds multi-objective models by adding the weighted sums of regularizers to the optimization criterion. BigARTM is known to combine well very different objectives, including sparsing, smoothing, topics decorrelation and many others. Such combination of regularizers significantly improves several quality measures at once almost without any loss of the perplexity.

### References

* Vorontsov K., Frei O., Apishev M., Romov P., Dudarenko M. BigARTM: [Open Source Library for Regularized Multimodal Topic Modeling of Large Collections](https://s3-eu-west-1.amazonaws.com/artm/Voron15aist.pdf) //  Analysis of Images, Social Networks and Texts. 2015.
* Vorontsov K., Frei O., Apishev M., Romov P., Dudarenko M., Yanina A. [Non-Bayesian Additive Regularization for Multimodal Topic Modeling of Large Collections](https://s3-eu-west-1.amazonaws.com/artm/Voron15cikm-tm.pdf) // Proceedings of the 2015 Workshop on Topic Models: Post-Processing and Applications, October 19, 2015 - pp. 29-37.
* Vorontsov K., Potapenko A., Plavin A. [Additive Regularization of Topic Models for Topic Selection and Sparse Factorization.](https://s3-eu-west-1.amazonaws.com/artm/voron15slds.pdf) // Statistical Learning and Data Sciences. 2015 — pp. 193-202.
* Vorontsov K. V., Potapenko A. A. [Additive Regularization of Topic Models](https://s3-eu-west-1.amazonaws.com/artm/voron-potap14artm-eng.pdf) // Machine Learning Journal, Special Issue “Data Analysis and Intelligent Optimization”, Springer, 2014.
* More publications can be found in our [wiki page](https://github.com/bigartm/bigartm/wiki/Publications).

### Related Software Packages

- [David Blei's List](http://www.cs.columbia.edu/~blei/topicmodeling_software.html) of Open Source topic modeling software
- [MALLET](http://mallet.cs.umass.edu/topics.php): Java-based toolkit for language processing with topic modeling package
- [Gensim](https://radimrehurek.com/gensim/): Python topic modeling library
- [Vowpal Wabbit](https://github.com/JohnLangford/vowpal_wabbit) has an implementation of [Online-LDA algorithm](https://github.com/JohnLangford/vowpal_wabbit/wiki/Latent-Dirichlet-Allocation)


# How to Use

### Installing

Download [binary release](https://github.com/bigartm/bigartm/releases) or build from source using cmake:
```bash
$ mkdir build && cd build
$ cmake ..
$ make install
```

### Command-line interface

Check out [documentation for `bigartm`](http://docs.bigartm.org/en/latest/tutorials/bigartm_cli.html).

Examples:

* Basic model (20 topics, outputed to CSV-file, inferred in 10 passes)

```bash
bigartm.exe -d docword.kos.txt -v vocab.kos.txt --write-model-readable model.txt
--passes 10 --batch-size 50 --topics 20
```

* Basic model with less tokens (filtered extreme values based on token's frequency)
```bash
bigartm.exe -d docword.kos.txt -v vocab.kos.txt --dictionary-max-df 50% --dictionary-min-df 2
--passes 10 --batch-size 50 --topics 20 --write-model-readable model.txt
```

* Simple regularized model (increase sparsity up to 60-70%)
```bash
bigartm.exe -d docword.kos.txt -v vocab.kos.txt --dictionary-max-df 50% --dictionary-min-df 2
--passes 10 --batch-size 50 --topics 20  --write-model-readable model.txt 
--regularizer "0.05 SparsePhi" "0.05 SparseTheta"
```

* More advanced regularize model, with 10 sparse objective topics, and 2 smooth background topics
```bash
bigartm.exe -d docword.kos.txt -v vocab.kos.txt --dictionary-max-df 50% --dictionary-min-df 2
--passes 10 --batch-size 50 --topics obj:10;background:2 --write-model-readable model.txt
--regularizer "0.05 SparsePhi #obj"
--regularizer "0.05 SparseTheta #obj"
--regularizer "0.25 SmoothPhi #background"
--regularizer "0.25 SmoothTheta #background" 
```

### Interactive Python interface

BigARTM supports full-featured and clear Python API (see [Installation](http://docs.bigartm.org/en/latest/installation/index.html) to configure Python API for your OS).

Example:

```python
import artm

# Prepare data
# Case 1: data in CountVectorizer format
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from numpy import array

cv = CountVectorizer(max_features=1000, stop_words='english')
n_wd = array(cv.fit_transform(fetch_20newsgroups().data).todense()).T
vocabulary = cv.get_feature_names()

bv = artm.BatchVectorizer(data_format='bow_n_wd',
                          n_wd=n_wd,
                          vocabulary=vocabulary)

# Case 2: data in UCI format (https://archive.ics.uci.edu/ml/datasets/Bag+of+Words)
bv = artm.BatchVectorizer(data_format='bow_uci',
                          collection_name='kos',
                          target_folder='kos_batches')

# Learn simple LDA model (or you can use advanced artm.ARTM)
model = artm.LDA(num_topics=15, dictionary=bv.dictionary)
model.fit_offline(bv, num_collection_passes=20)

# Print results
model.get_top_tokens()
```

Refer to [tutorials](http://docs.bigartm.org/en/latest/tutorials/python_tutorial.html) for details on how to start using BigARTM from Python, [user's guide](http://docs.bigartm.org/en/latest/tutorials/python_userguide/index.html) can provide information about more advanced features and cases.

### Low-level API

  - [C++ Interface](http://docs.bigartm.org/en/latest/api_references/cpp_interface.html)
  - [Plain C Interface](http://docs.bigartm.org/en/latest/api_references/c_interface.html)


## Contributing

Refer to the [Developer's Guide](http://docs.bigartm.org/en/latest/devguide.html) and follows [Code Style](https://github.com/bigartm/bigartm/wiki/Code-style).

To report a bug use [issue tracker](https://github.com/bigartm/bigartm/issues). To ask a question use [our mailing list](https://groups.google.com/forum/#!forum/bigartm-users). Feel free to make [pull request](https://github.com/bigartm/bigartm/pulls).


## License

BigARTM is released under [New BSD License](https://raw.github.com/bigartm/bigartm/master/LICENSE) that allowes unlimited redistribution for any purpose (even for commercial use) as long as its copyright notices and the license’s disclaimers of warranty are maintained.
