# Contributors:

Zhenglei WANG

Jingyu ZHANG

Runjia JIANG

Xiaoman HU


# Description of the Implemented Classifier:

The implemented classifier follows a multi-step approach for sentiment classification:

## Preprocessing:

Text preprocessing involves identifying and extracting short sentences containing the target entity and removing punctuation. This is achieved through the preprocess_text method, which utilizes regular expressions to split sentences based on common discourse markers like conjunctions and punctuation.
Word Embedding:

For word embedding, the RoBERTa model is utilized. RoBERTa is a transformer-based language model pre-trained on vast amounts of textual data. The RobertaTokenizer is employed to tokenize the input text, and then the RoBERTa model (RobertaModel) is used to obtain embeddings for each tokenized input. The maximum length of tokens is set to 48 for padding and truncation.
Neural Feature Engineering:

The RoBERTa model serves as the basis for neural feature engineering. By passing the tokenized inputs through RoBERTa, a 768-dimensional feature vector is generated for each input sentence. These vectors capture rich contextual information from the text.
Classification:

For classification, a Support Vector Machine (SVM) classifier with radial basis function (RBF) kernel is employed. SVM is chosen for its robust performance in text classification tasks. The features obtained from the RoBERTa model are standardized using StandardScaler, and then used to train the SVM model.
Resources:

The model utilizes the RoBERTa pre-trained transformer model for word embedding.
SVM classifier with RBF kernel is used for sentiment classification.
The code utilizes the transformers library for RoBERTa tokenization and model loading, and scikit-learn for SVM implementation.
Accuracy on Dev Dataset:

The accuracy achieved on the development dataset is 
