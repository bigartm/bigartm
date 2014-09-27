# This example generates a synthetic in-memory collection with 100 items and 60 tokens.
# First 40 tokens are split on 10 clear topics according to this condition: "(token_id % 10) == (item_id % 10)"
# Last 20 tokens are randomly allocated to documents.
# The example demonstrates that in this clear situation BigARTM can precisely recover all topics.

import artm.messages_pb2, artm.library, random

# Generate small collection of random items
num_tokens = 60
num_items = 100
batch = artm.messages_pb2.Batch()
for token_id in range(0, num_tokens):
  batch.token.append('token' + str(token_id))

for item_id in range(0, num_items):
  item = batch.item.add()
  item.id = item_id
  field = item.field.add()
  for token_id in range(0, num_tokens):
    field.token_id.append(token_id)
    background_count = random.randint(1, 5) if (token_id >= 40) else 0
    topical_count    = 10 if ((token_id < 40) and ((token_id % 10) == (item_id % 10))) else 0
    field.token_count.append(background_count + topical_count)

# Create master component and infer topic model
with artm.library.MasterComponent() as master:
  master.AddBatch(batch)
  perplexity_score = master.CreatePerplexityScore()
  top_tokens_score = master.CreateTopTokensScore(num_tokens = 4)
  model = master.CreateModel(topics_count = 10, inner_iterations_count = 10)
  model.EnableScore(perplexity_score)
  model.EnableScore(top_tokens_score)

  for iter in range(0, 10):
    master.InvokeIteration(1)        # Invoke one scan of the entire collection...
    master.WaitIdle();               # and wait until it completes.
    model.Synchronize();             # Synchronize topic model.
    print "Iter#" + str(iter) + ": Perplexity = %.3f" % perplexity_score.GetValue(model).value

  top_tokens = top_tokens_score.GetValue(model)

artm.library.Visualizers.PrintTopTokensScore(top_tokens)
