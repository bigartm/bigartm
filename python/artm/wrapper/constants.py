"""
Constants values, used in messages
"""

# TODO: collect constants automatically

Stream_Type_Global = 0
Stream_Type_ItemIdModulus = 1
RegularizerConfig_Type_SmoothSparseTheta = 0
RegularizerConfig_Type_SmoothSparsePhi = 1
RegularizerConfig_Type_DecorrelatorPhi = 2
RegularizerConfig_Type_LabelRegularizationPhi = 4
RegularizerConfig_Type_SpecifiedSparsePhi = 5
RegularizerConfig_Type_ImproveCoherencePhi = 6
ScoreConfig_Type_Perplexity = 0
ScoreData_Type_Perplexity = 0
ScoreConfig_Type_SparsityTheta = 1
ScoreData_Type_SparsityTheta = 1
ScoreConfig_Type_SparsityPhi = 2
ScoreData_Type_SparsityPhi = 2
ScoreConfig_Type_ItemsProcessed = 3
ScoreData_Type_ItemsProcessed = 3
ScoreConfig_Type_TopTokens = 4
ScoreData_Type_TopTokens = 4
ScoreConfig_Type_ThetaSnippet = 5
ScoreData_Type_ThetaSnippet = 5
ScoreConfig_Type_TopicKernel = 6
ScoreData_Type_TopicKernel = 6
PerplexityScoreConfig_Type_UnigramDocumentModel = 0
PerplexityScoreConfig_Type_UnigramCollectionModel = 1
CollectionParserConfig_Format_BagOfWordsUci = 0
CollectionParserConfig_Format_MatrixMarket = 1
CollectionParserConfig_Format_VowpalWabbit = 2
CollectionParserConfig_Format_Cooccurrence = 3
GetTopicModelArgs_RequestType_Pwt = 0
GetTopicModelArgs_RequestType_Nwt = 1
GetTopicModelArgs_RequestType_TopicNames = 2
GetTopicModelArgs_RequestType_Tokens = 3
InitializeModelArgs_SourceType_Dictionary = 0
InitializeModelArgs_SourceType_Batches = 1
SpecifiedSparsePhiConfig_Mode_SparseTopics = 0
SpecifiedSparsePhiConfig_Mode_SparseTokens = 1
ProcessBatchesArgs_ThetaMatrixType_None = 0
ProcessBatchesArgs_ThetaMatrixType_Dense = 1
ProcessBatchesArgs_ThetaMatrixType_Sparse = 2
ProcessBatchesArgs_ThetaMatrixType_Cache = 3
ProcessBatchesArgs_ThetaMatrixType_External = 4
CopyRequestResultArgs_RequestType_GetThetaSecondPass = 0
CopyRequestResultArgs_RequestType_GetModelSecondPass = 1
TopicModel_OperationType_Initialize = 0
TopicModel_OperationType_Increment = 1
TopicModel_OperationType_Overwrite = 2
TopicModel_OperationType_Remove = 3
TopicModel_OperationType_Ignore = 4
GetTopicModelArgs_MatrixLayout_Dense = 0
GetTopicModelArgs_MatrixLayout_Sparse = 1
GetTopicModelArgs_MatrixLayout_External = 2
GetThetaMatrixArgs_MatrixLayout_Dense = 0
GetThetaMatrixArgs_MatrixLayout_Sparse = 1
GetThetaMatrixArgs_MatrixLayout_External = 2
