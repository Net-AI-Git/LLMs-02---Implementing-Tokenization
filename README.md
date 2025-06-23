# 🔤 NLP Tokenization Techniques Implementation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.8+-green.svg)](https://www.nltk.org/)
[![spaCy](https://img.shields.io/badge/spaCy-3.7+-09a3d5.svg)](https://spacy.io/)
[![Transformers](https://img.shields.io/badge/Transformers-4.42.1-yellow.svg)](https://huggingface.co/transformers/)

## 📋 Description

A comprehensive exploration and comparison of various tokenization techniques in Natural Language Processing. This project demonstrates the implementation and performance analysis of different tokenizers including NLTK, spaCy, BERT, and XLNet, providing insights into their unique approaches to text segmentation and vocabulary building.

## 🚀 Features

- 📊 **Multi-Tokenizer Comparison** - Side-by-side implementation of 4 major tokenization approaches
- ⚡ **Performance Benchmarking** - Execution time measurement for each tokenizer
- 🔧 **Custom Vocabulary Building** - Dynamic vocabulary construction from datasets using TorchText
- 📈 **Token Frequency Analysis** - Statistical analysis of tokenization patterns
- 🏷️ **Special Token Handling** - Implementation of BOS, EOS, PAD, and UNK tokens
- 🔍 **Subword Tokenization** - Demonstration of WordPiece (BERT) and SentencePiece (XLNet) methods

## 🛠️ Technologies Used

- **Python 3.8+**
- **NLTK** - Natural Language Toolkit for basic tokenization
- **spaCy** - Industrial-strength NLP with linguistic features
- **Transformers (HuggingFace)** - BERT and XLNet tokenizers
- **TorchText** - PyTorch's text processing utilities
- **NumPy** - Numerical computing
- **scikit-learn** - Machine learning utilities

## 📦 Installation

Since this project is developed in Google Colab, all dependencies are installed within the notebook. To run locally:

```bash
# Clone the repository
git clone https://github.com/Net-AI-Git/NLP-Tokenization-Techniques.git
cd NLP-Tokenization-Techniques

# Install required packages
pip install nltk==3.8.1
pip install transformers==4.42.1
pip install sentencepiece
pip install spacy==3.7.2
pip install torch==2.2.2
pip install torchtext==0.17.2
pip install numpy==1.26.0
pip install scikit-learn

# Download spaCy language models
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

## 💻 Usage

### Basic Tokenization Example

```python
from nltk.tokenize import word_tokenize

text = "Natural language processing helps computers understand human communication."
tokens = word_tokenize(text)
print(tokens)
# Output: ['Natural', 'language', 'processing', 'helps', 'computers', 'understand', 'human', 'communication', '.']
```

### BERT Tokenization with Subwords

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize("BERT uses bidirectional attention mechanisms.")
print(tokens)
# Output: ['bert', 'uses', 'bid', '##ire', '##ction', '##al', 'attention', 'mechanisms', '.']
```

### Building Custom Vocabulary

```python
from torchtext.vocab import build_vocab_from_iterator

vocab = build_vocab_from_iterator(yield_tokens(dataset), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
```

## 📊 Results

The project demonstrates comparative analysis of different tokenizers:

**[INSERT PERFORMANCE COMPARISON CHART HERE]**
*Location: Add a bar chart showing execution times for each tokenizer*

**[INSERT TOKEN FREQUENCY DISTRIBUTION HERE]**
*Location: Add a visualization of token frequencies across different methods*

Key findings:
- NLTK provides fast, rule-based tokenization suitable for general purposes
- spaCy offers linguistic features alongside tokenization
- BERT's WordPiece handles out-of-vocabulary words through subword units
- XLNet's SentencePiece provides language-independent tokenization

## 🔮 Future Work

- 🌍 Implement multilingual tokenization comparison
- 📱 Add support for social media text tokenization
- 🧮 Include BPE (Byte Pair Encoding) tokenizer implementation
- 📊 Expand benchmarking to include memory usage metrics
- 🔄 Add real-time tokenization API endpoint

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 👤 Author

**Netanel Itzhak**
- LinkedIn: [www.linkedin.com/in/netanelitzhak](https://www.linkedin.com/in/netanelitzhak)
- GitHub: [Net-AI-Git](https://github.com/Net-AI-Git)
- Email: ntitz19@gmail.com

---

<p align="center">Made with ❤️ for the NLP community</p>
