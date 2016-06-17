<p align="center">
	<img alt="BigARTM Logo" src="http://bigartm.org/img/BigARTM-logo.svg" width="250">
</p>

The state-of-the-art platform for topic modeling.

[![Build Status](https://secure.travis-ci.org/bigartm/bigartm.png)](https://travis-ci.org/bigartm/bigartm)
[![GitHub license](https://img.shields.io/badge/license-New%20BSD-blue.svg)](https://raw.github.com/bigartm/bigartm/master/LICENSE.txt)

  - [Full Documentation](http://docs.bigartm.org/)
  - [User Mailing List](https://groups.google.com/forum/#!forum/bigartm-users)
  - [Download Releases](https://github.com/bigartm/bigartm/releases)
  - [User survey](http://goo.gl/forms/tr5EsPMcL2)


# What is BigARTM?

BigARTM is a tool for [topic modeling](https://www.cs.princeton.edu/~blei/papers/Blei2012.pdf) based on a novel technique called Additive Regularization of Topic Models. This technique effectively builds multi-objective models by adding the weighted sums of regularizers to the optimization criterion. BigARTM is known to combine well very different objectives, including sparsing, smoothing, topics decorrelation and many others. Such combinations of regularizers significantly improves several quality measures at once almost without any loss of the perplexity.

### References

* Vorontsov K., Frei O., Apishev M., Romov P., Dudarenko M. BigARTM: [Open Source Library for Regularized Multimodal Topic Modeling of Large Collections](https://s3-eu-west-1.amazonaws.com/artm/Voron15aist.pdf) //  Analysis of Images, Social Networks and Texts. 2015.
* Vorontsov K., Frei O., Apishev M., Romov P., Dudarenko M., Yanina A. [Non-Bayesian Additive Regularization for Multimodal Topic Modeling of Large Collections](https://s3-eu-west-1.amazonaws.com/artm/Voron15cikm-tm.pdf) // Proceedings of the 2015 Workshop on Topic Models: Post-Processing and Applications, October 19, 2015 - pp. 29-37.
* Vorontsov K., Potapenko A., Plavin A. [Additive Regularization of Topic Models for Topic Selection and Sparse Factorization.](https://s3-eu-west-1.amazonaws.com/artm/voron15slds.pdf) // Statistical Learning and Data Sciences. 2015 — pp. 193-202.
* Vorontsov K. V., Potapenko A. A. [Additive Regularization of Topic Models](https://s3-eu-west-1.amazonaws.com/artm/voron-potap14artm-eng.pdf) // Machine Learning Journal, Special Issue “Data Analysis and Intelligent Optimization”, Springer, 2014.
* More publications can be found in our [wiki page](https://github.com/bigartm/bigartm/wiki/Publications).

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
$ make install
```

### Command-line interface

Check out [documentation for `bigartm`](http://docs.bigartm.org/en/latest/tutorials/bigartm_cli.html).

Examples:

* Basic model (20 topics, outputed to CSV-file, infered in 10 passes)

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

Check out the documentation for the ARTM Python interface 
[in English](http://nbviewer.ipython.org/github/bigartm/bigartm-book/blob/master/ARTM_tutorial_EN.ipynb) and
[in Russian](http://nbviewer.ipython.org/github/bigartm/bigartm-book/blob/master/ARTM_tutorial_RU.ipynb) 

Refer to [tutorials](http://docs.bigartm.org/en/latest/tutorials/index.html) for details on how to install and start using Python interface.

```python
# A stub
import bigartm

model = bigartm.ARTM(num_topics=15
batch_vectorizer = artm.BatchVectorizer(data_format='bow_uci',
                                        collection_name='kos',
                                        target_folder='kos'))
model.fit_offline(batches, passes=5)
print model.phi_
```

### Low-level API

  - [C++ Interface](http://docs.bigartm.org/en/latest/ref/cpp_interface.html)
  - [Plain C Interface](http://docs.bigartm.org/en/latest/ref/c_interface.html)


## Contributing

Refer to the [Developer's Guide](http://docs.bigartm.org/en/latest/devguide.html).

To report a bug use [issue tracker](https://github.com/bigartm/bigartm/issues). To ask a question use [our mailing list](https://groups.google.com/forum/#!forum/bigartm-users). Feel free to make [pull request](https://github.com/bigartm/bigartm/pulls).


## License

BigARTM is released under [New BSD License](https://raw.github.com/bigartm/bigartm/master/LICENSE) that allowes unlimited redistribution for any purpose (even for commercial use) as long as its copyright notices and the license’s disclaimers of warranty are maintained.
