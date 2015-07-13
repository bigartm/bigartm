# This example demonstrates how to use AttachModel to manually manipulate values in topic model.


import artm.messages_pb2, artm.library, sys, glob

# Parse collection
data_folder = sys.argv[1] if (len(sys.argv) >= 2) else ''
target_folder = 'kos'
collection_name = 'kos'
if not glob.glob(target_folder + "/*.batch"):
    artm.library.Library().ParseCollection(
        docword_file_path=data_folder + 'docword.' + collection_name + '.txt',
        vocab_file_path=data_folder + 'vocab.' + collection_name + '.txt',
        target_folder=target_folder)

# Find file names of all batches in target folder
batches = glob.glob(target_folder + "/*.batch")

# Create master component
with artm.library.MasterComponent() as master:
    theta_snippet_score = master.CreateThetaSnippetScore()

    # Initialize model
    pwt_model = "pwt"
    master.InitializeModel(pwt_model, batch_folder=target_folder, topics_count=10)
    topic_model, numpy_matrix = master.AttachModel(pwt_model)
    numpy_matrix[:, 4] = 0

    # Perform iterations
    for iteration in range(0, 5):
        master.ProcessBatches(pwt_model, batches, "nwt")
        master.NormalizeModel("nwt", pwt_model)

    # Note that 5th topic is fully zero; this is because we performed "numpy_matrix[:, 4] = 0".
    artm.library.Visualizers.PrintThetaSnippetScore(theta_snippet_score.GetValue(pwt_model))
