# Contributors:

Zhenglei WANG

Jingyu ZHANG

Runjia JIANG

Xiaoman HU


# Description of the Implemented Classifier:

The implemented classifier follows a multi-step approach for sentiment classification:

## Preprocessing:

Text preprocessing involves identifying and extracting short sentences containing the target entity and removing punctuation. This is achieved through the 'preprocess_text' method, which utilizes regular expressions to split sentences based on common discourse markers like conjunctions and punctuation.
The following steps are performed:

Text segmentation: Since a sentence may contain multiple emotional subjects, it is common to use punctuation or discourse markers to separate them. The preprocessing method splits the original text based on punctuation or discourse markers like conjunctions, retaining only the short sentences containing the target entity.

Target inclusion: Only the short sentences containing the target entity are retained for further processing. The target entity is added to the beginning of the sentence to ensure context preservation.

Punctuation removal: Punctuation marks are removed from the retained sentences.

## Word Embedding:
 
In this step, we utilize the pre-trained RoBERTa model from the transformer architecture to perform word embedding. Here's a breakdown of the process:

Tokenization:

We start by tokenizing the input text using RoBERTa's tokenizer. This breaks the text into individual tokens, facilitating model comprehension.

Padding and Truncation:

Since RoBERTa requires fixed-length inputs, we pad all token sequences to the maximum length (48 tokens) using padding='max_length'. If the text exceeds 48 tokens, it's truncated.

Tensorization:

Next, we convert the token sequences into PyTorch tensors for processing within PyTorch.

Feature Extraction:

We then feed the tokenized and padded text into the RoBERTa model to obtain hidden state representations for each token. These hidden states capture the semantic information of each token.

Summarization:

To obtain a single vector representation for the entire sentence, we extract the hidden state of the CLS token from the last layer. This CLS token representation serves as the vector representation for the entire sentence.

Return:

Finally, we convert this sentence vector representation into a NumPy array for further processing.

Through this word embedding step, we transform text data into numerical representations, which can be used as input for machine learning models. In the data_transform method, we store these word embedding features in the 'features' column of the DataFrame for subsequent training or prediction tasks.

## Neural Feature Engineering:

The RoBERTa model is utilized for neural feature engineering, enabling the capture of rich contextual information from the text. Here's a concise explanation of the key steps:

## Tokenization and Embedding:
The input text undergoes tokenization using RoBERTa's tokenizer. Tokens are then processed through the RoBERTa model to generate 768-dimensional feature vectors for each sentence. This process captures detailed contextual information crucial for sentiment understanding.

## Feature Representation:
Each 768-dimensional feature vector encapsulates the semantic content of a sentence, including word arrangement, syntactic relationships, and overall context. These vectors serve as the neural features utilized for sentiment classification.

## Classification:

For classification, a Support Vector Machine (SVM) classifier with radial basis function (RBF) kernel is employed. SVM is chosen for its robust performance in text classification tasks. The features obtained from the RoBERTa model are standardized using StandardScaler, and then used to train the SVM model.

## Resources:

The classifier utilizes the following resources:

Pre-trained RoBERTa transformer model for word embedding.
Support Vector Machine (SVM) classifier with a radial basis function (RBF) kernel for sentiment classification.
The 'transformers' library for RoBERTa tokenization and model loading.
The 'scikit-learn' library for SVM implementation.

## Accuracy on Dev Dataset:

The accuracy achieved on the devdata is 0.88.
