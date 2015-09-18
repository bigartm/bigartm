Compare BigARTM against other libraries
=======================================

OnlineLDA Experiment
--------------------

Compare performance of running Online-LDA algorithm [1].

Competitors:

  - BigARTM with Smoothing regularizer
  - [LDA mode in Vowpal Wabbit](https://github.com/JohnLangford/vowpal_wabbit/wiki/Latent-Dirichlet-Allocation)
  - [Gensim LDAMulticore](http://radimrehurek.com/gensim/models/ldamulticore.html)

How To Use
----------

Use module `onlinelda` to write script that runs BigARTM/VW/Gensim in necessary settings, see `example_run.py`.