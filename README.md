# 🔤 LLMs-02: Implementing Tokenization

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Colab](https://img.shields.io/badge/Google-Colab-orange.svg)](https://colab.research.google.com/)
[![NLTK](https://img.shields.io/badge/NLTK-3.9%2B-green.svg)](https://www.nltk.org/)
[![spaCy](https://img.shields.io/badge/spaCy-3.8%2B-blue.svg)](https://spacy.io/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.42.1-yellow.svg)](https://huggingface.co/transformers/)

A comprehensive exploration of tokenization techniques in Natural Language Processing, demonstrating various approaches from classical methods to modern transformer-based tokenizers.

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [✨ Features](#-features)
- [🛠️ Technologies Used](#️-technologies-used)
- [⚙️ Installation](#️-installation)
- [🚀 Usage](#-usage)
- [📁 Project Structure](#-project-structure)
- [🔮 Future Work](#-future-work)
- [🙏 Acknowledgments](#-acknowledgments)
- [📬 Contact](#-contact)

## 🎯 Overview

This project demonstrates various tokenization techniques essential for NLP applications. It covers traditional rule-based tokenizers, modern subword tokenization methods, and transformer-based approaches, providing hands-on examples and performance comparisons across different methodologies.

## ✨ Features

🔍 **Multiple Tokenization Methods**
- NLTK word tokenization
- spaCy linguistic tokenization  
- BERT WordPiece tokenization
- XLNet SentencePiece tokenization
- TorchText basic tokenization

📊 **Comprehensive Analysis**
- Token frequency analysis
- Performance timing comparisons
- Special token handling demonstrations
- Vocabulary building from scratch

🛠️ **Practical Implementation**
- Custom vocabulary creation
- Token-to-ID mapping
- Padding and special tokens handling
- Unknown word management

## 🛠️ Technologies Used

- **Python 3.8+**
- **NLTK 3.9+** - Classical NLP tokenization
- **spaCy 3.8+** - Advanced linguistic processing
- **🤗 Transformers 4.42.1** - Modern transformer tokenizers
- **PyTorch 2.2.2** - Deep learning framework
- **TorchText 0.17.2** - Text preprocessing utilities
- **scikit-learn** - Machine learning utilities
- **NumPy** - Numerical computing

## ⚙️ Installation

### Google Colab (Recommended)

1. **Open the notebook directly in Google Colab**
   - Click on the notebook file `Implementing_Tokenization (1).ipynb`
   - Select "Open in Colab" or upload to your Google Drive

2. **Run the installation cell**
   - The notebook includes all necessary pip install commands
   - All dependencies will be installed automatically
   - No manual setup required

### Local Setup (Optional)

If you prefer to run locally:

1. **Clone the repository**
```bash
git clone https://github.com/Net-AI-Git/LLMs-02---Implementing-Tokenization.git
cd LLMs-02---Implementing-Tokenization
```

2. **Install dependencies** (as shown in the notebook)
```bash
pip install nltk transformers==4.42.1 sentencepiece spacy scikit-learn torch==2.2.2 torchtext==0.17.2 numpy==1.26.0
```

3. **Download language models**
```bash
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

## 🚀 Usage

### Running the Notebook

1. Open the notebook in Google Colab
2. Execute the first cell to install all dependencies
3. Run cells sequentially to explore different tokenization methods

### Basic Tokenization Example

```python
import nltk
from transformers import BertTokenizer

# NLTK tokenization
text = "Natural language processing helps computers understand human communication."
tokens = nltk.word_tokenize(text)
print(tokens)
# Output: ['Natural', 'language', 'processing', 'helps', 'computers', 'understand', 'human', 'communication', '.']

# BERT tokenization
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_tokens = tokenizer.tokenize(text)
print(bert_tokens)
# Output: ['natural', 'language', 'processing', 'helps', 'computers', 'understand', 'human', 'communication', '.']
```

### Building Custom Vocabulary

```python
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

tokenizer = get_tokenizer("basic_english")
vocab = build_vocab_from_iterator(yield_tokens(dataset), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
```

### Key Demonstrations

The notebook covers:
- **Tokenization Comparison**: Side-by-side analysis of NLTK, spaCy, BERT, and XLNet tokenizers
- **Performance Timing**: Speed comparison across different methods
- **Vocabulary Building**: Creating custom vocabularies with TorchText
- **Special Token Handling**: Working with `<bos>`, `<eos>`, `<pad>`, and `<unk>` tokens
- **Frequency Analysis**: Token distribution and frequency counting

## 📁 Project Structure

```
LLMs-02---Implementing-Tokenization/
│
├── Implementing_Tokenization (1).ipynb  # Main notebook with all implementations
└── README.md                           # Project documentation
```

## 📊 Key Findings

The analysis demonstrates distinct characteristics of each tokenization approach:

- **NLTK**: Fast rule-based tokenization, excellent for basic text processing and research
- **spaCy**: Comprehensive linguistic analysis with POS tagging, dependency parsing, and NER
- **BERT**: Subword tokenization using WordPiece, effective for handling out-of-vocabulary words  
- **XLNet**: SentencePiece tokenization with superior handling of morphologically rich languages

**Performance Insights:**
- Classical tokenizers (NLTK, spaCy) maintain word boundaries more strictly
- Subword tokenizers (BERT, XLNet) produce more granular representations
- Different approaches to handling punctuation and contractions
- Trade-offs between speed and linguistic sophistication

## 🔮 Future Work

- 🌐 **Multilingual Support**: Extend analysis to non-English languages (German, Arabic, Chinese)
- ⚡ **GPU Acceleration**: Implement CUDA-optimized tokenization for large datasets
- 🏗️ **Custom Tokenizer**: Build domain-specific tokenization strategies for technical texts
- 📈 **Downstream Evaluation**: Test tokenization impact on classification and NER tasks
- 🔄 **Streaming Processing**: Implement real-time tokenization for large text streams
- 📚 **Byte-Pair Encoding**: Add BPE tokenization examples and comparisons

## 🙏 Acknowledgments

- Hugging Face team for the excellent Transformers library and pre-trained models
- spaCy developers for robust linguistic processing tools and language models
- NLTK contributors for foundational NLP utilities and datasets
- PyTorch team for the flexible deep learning framework
- Google Colab for providing accessible computational resources

---

## 📬 Contact

**Netanel Itzhak**
- 💼 LinkedIn: [www.linkedin.com/in/netanelitzhak](https://www.linkedin.com/in/netanelitzhak)
- 📧 Email: ntitz19@gmail.com
- 🐙 GitHub: [Net-AI-Git](https://github.com/Net-AI-Git)

---

*This project serves as an educational exploration of tokenization techniques in NLP. The implementations demonstrate practical applications of various tokenization approaches and their use cases in modern NLP pipelines.*
