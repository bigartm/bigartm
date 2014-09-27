import artm.library, artm.messages_pb2
import operator, random, glob, sys

# Some configuration numbers
batch_size = 500
processors_count = 4
eps = 1e-100
limit_collection_size = 50000 # don't load more that this docs
topics_count = 20
outer_iteration_count = 10
inner_iterations_count = 10
top_tokens_count_to_visualize = 4

vocab_file = 'vocab.kos.txt'
docword_file = 'docword.kos.txt'
target_folder = 'batches'
dictionary_file = 'kos.dictionary'

artm_library = artm.library.Library()

batches_found = len(glob.glob(target_folder + "/*.batch"))
if batches_found == 0:
    print "No batches found, parsing them from textual collection ",
    collection_parser_config = messages_pb2.CollectionParserConfig();
    collection_parser_config.format = CollectionParserConfig_Format_BagOfWordsUci

    data_folder = sys.argv[1] if (len(sys.argv) >= 2) else ""
    collection_parser_config.docword_file_path = data_folder + docword_file
    collection_parser_config.vocab_file_path = data_folder + vocab_file
    collection_parser_config.target_folder = target_folder
    collection_parser_config.dictionary_file_name = dictionary_file
    unique_tokens = artm_library.ParseCollection(collection_parser_config);
    print " OK."
else:
    print "Found " + str(batches_found) + " batches, using them."
    unique_tokens = artm_library.LoadDictionary(target_folder + "/" + dictionary_file);

master_config = artm.messages_pb2.MasterComponentConfig()
master_config.processors_count = processors_count
master_config.cache_theta = 1
master_config.disk_path = target_folder

with artm.library.MasterComponent(master_config) as master:
  perplexity_score     = master.CreatePerplexityScore()
  sparsity_theta_score = master.CreateSparsityThetaScore()
  sparsity_phi_score   = master.CreateSparsityPhiScore()
  top_tokens_score     = master.CreateTopTokensScore()
  theta_snippet_score  = master.CreateThetaSnippetScore()
  kernel_score         = master.CreateTopicKernelScore()

  regularizer_theta    = master_component.CreateDirichletThetaRegularizer()
  regularizer_decor    = master_component.CreateDecorrelatorPhiRegularizer()

  model = master_component.CreateModel(topics_count = topics_count, inner_iterations_count = inner_iterations_count)
  model.EnableScore(perplexity_score)
  model.EnableScore(sparsity_theta_score)
  model.EnableScore(sparsity_theta_score)


  model_config.score_name.append(perplexity_score_name)
  model_config.score_name.append(sparsity_theta_score_name)
  model_config.score_name.append(sparsity_phi_score_name)
  model_config.score_name.append(topic_kernel_score_name)

  
  initial_topic_model = artm.messages_pb2.TopicModel();
  initial_topic_model.topics_count = topics_count;
  initial_topic_model.name = model.name()

  rnd = random.Random()
  rnd.seed(123)
  model.Initialize(unique_tokens, rnd)

  for iter in range(0, outer_iteration_count):
      master_component.InvokeIteration(1)
      master_component.WaitIdle(120000);
      model.Synchronize(0.0)

      topic_model = master_component.GetTopicModel(model)
      perplexity_score = master_component.GetScore(model, perplexity_score_name)
      sparsity_theta_score = master_component.GetScore(model, sparsity_theta_score_name)
      sparsity_phi_score = master_component.GetScore(model, sparsity_phi_score_name)
      topic_kernel_score = master_component.GetScore(model, topic_kernel_score_name)

      print "Iter# = " + str(iter) + \
              ", Perplexity = " + str(perplexity_score.value) + \
              ", SparsityTheta = " + str(sparsity_theta_score.value) +\
              ", SparsityPhi = " + str(sparsity_phi_score.value) +\
              ", KernelSize = " + str(topic_kernel_score.average_kernel_size) +\
              ", KernelPurity = " + str(topic_kernel_score.average_kernel_purity) +\
                ", KernelContrast = " + str(topic_kernel_score.average_kernel_contrast)

  # Log to 7 words in each topic
  tokens_size = len(topic_model.token)
  topics_size = topic_model.topics_count

  for topic_index in range(0, topics_size):
      token_map = {}
      best_tokens = '#' + str(topic_index + 1) + ': '
      for token_index in range(0, tokens_size):
          token = topic_model.token[token_index];
          token_weight = topic_model.token_weights[token_index].value[topic_index]
          token_map[token] = token_weight
      sorted_token_map = sorted(token_map.iteritems(), key=operator.itemgetter(1), reverse=True)
      for best_token in range(0, top_tokens_count_to_visualize):
          best_tokens = best_tokens + sorted_token_map[best_token][0] + ', '
      print best_tokens.rstrip(', ')

  docs_to_show = 7
  print "\nThetaMatrix (first " + str(docs_to_show) + " documents):"
  theta_matrix = master_component.GetThetaMatrix(model)
  for j in range(0, topics_size):
      print "Topic" + str(j) + ": ",
      for i in range(0, min(docs_to_show, len(theta_matrix.item_id))):
          weight = theta_matrix.item_weights[i].value[j]
          print "%.3f\t" % weight,
      print "\n",

  print 'Done with regularization!'
