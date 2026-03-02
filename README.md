# 🤖 AI Toolkit

A collection of Artificial Intelligence utilities and algorithms implemented in Python — covering core ML concepts, neural network helpers, NLP preprocessing, and data tools.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/Tests-passing-brightgreen)

---

## 📋 Overview

This project provides clean, well-tested implementations of common AI/ML building blocks:

- 🧠 **Neural Networks** — activation functions, weight initialization, forward pass
- 📊 **Data Preprocessing** — normalization, train/test split, feature scaling
- 🔤 **NLP Utilities** — tokenization, TF-IDF, cosine similarity
- 📈 **Evaluation Metrics** — accuracy, precision, recall, F1-score, confusion matrix
- 🔍 **Search Algorithms** — BFS, DFS, A* pathfinding

---

## 🛠️ Tech Stack

- Python 3.9+
- NumPy
- pytest (testing)

---

## 📦 Installation

```bash
git clone https://github.com/your-username/ai-toolkit.git
cd ai-toolkit
pip install -r requirements.txt
```

---

## 🚀 Usage

```python
from src.neural import sigmoid, relu, softmax
from src.metrics import accuracy, f1_score
from src.nlp import tokenize, cosine_similarity
from src.preprocessing import normalize, train_test_split

# Activation functions
print(sigmoid(0))          # 0.5
print(relu(-3))            # 0.0

# Evaluate a model
y_true = [1, 0, 1, 1, 0]
y_pred = [1, 0, 0, 1, 0]
print(accuracy(y_true, y_pred))   # 0.8
print(f1_score(y_true, y_pred))   # 0.857

# NLP
tokens = tokenize("Artificial Intelligence is amazing!")
sim = cosine_similarity([1, 0, 1], [1, 1, 0])
```

---

## 📁 Project Structure

```
ai-toolkit/
├── src/
│   ├── __init__.py
│   ├── neural.py          # Activation functions & NN helpers
│   ├── metrics.py         # Evaluation metrics
│   ├── nlp.py             # NLP utilities
│   ├── preprocessing.py   # Data preprocessing
│   └── search.py          # Search algorithms (BFS, DFS, A*)
├── tests/
│   ├── test_neural.py
│   ├── test_metrics.py
│   ├── test_nlp.py
│   └── test_preprocessing.py
├── docs/
│   └── API.md
├── .github/
│   └── workflows/
│       └── tests.yml      # CI/CD with GitHub Actions
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🧪 Running Tests

```bash
pytest                        # Run all tests
pytest --cov=src              # With coverage report
pytest tests/test_neural.py   # Single file
```

---

## 🤝 Contributing

1. Fork the repo
2. Create a branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'feat: add new algorithm'`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

Please follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages.

---

## 📄 License

MIT © [Your Name]
