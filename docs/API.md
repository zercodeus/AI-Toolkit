# 📖 API Reference

## `src.neural`

| Function | Description |
|---|---|
| `sigmoid(x)` | Sigmoid activation → (0, 1) |
| `relu(x)` | ReLU activation → max(0, x) |
| `tanh(x)` | Hyperbolic tangent → (-1, 1) |
| `softmax(values)` | Converts logits to probabilities |
| `initialize_weights(n_in, n_out, method)` | Xavier / He / zeros init |
| `forward_pass(inputs, weights, bias, activation)` | Single-neuron forward pass |

---

## `src.metrics`

| Function | Description |
|---|---|
| `accuracy(y_true, y_pred)` | Fraction of correct predictions |
| `precision(y_true, y_pred)` | TP / (TP + FP) |
| `recall(y_true, y_pred)` | TP / (TP + FN) |
| `f1_score(y_true, y_pred)` | Harmonic mean of precision & recall |
| `confusion_matrix(y_true, y_pred)` | Returns dict {tp, fp, tn, fn} |
| `mean_squared_error(y_true, y_pred)` | MSE for regression |
| `mean_absolute_error(y_true, y_pred)` | MAE for regression |

---

## `src.nlp`

| Function | Description |
|---|---|
| `tokenize(text, lowercase)` | Word tokenizer, removes punctuation |
| `remove_stopwords(tokens, stopwords)` | Filters out common words |
| `term_frequency(tokens)` | Normalized TF dict |
| `inverse_document_frequency(corpus)` | IDF dict for a corpus |
| `tfidf(tokens, corpus)` | TF-IDF scores |
| `cosine_similarity(vec_a, vec_b)` | Similarity score in [0, 1] |

---

## `src.preprocessing`

| Function | Description |
|---|---|
| `normalize(data)` | Min-max scaling to [0, 1] |
| `standardize(data)` | Z-score normalization |
| `train_test_split(data, test_size, shuffle, seed)` | Split dataset |
| `one_hot_encode(labels)` | Returns (encoded, classes) |
| `flatten(nested)` | Recursively flatten a list |

---

## `src.search`

| Function | Description |
|---|---|
| `bfs(graph, start, goal)` | Breadth-First Search |
| `dfs(graph, start, goal)` | Depth-First Search |
| `astar(graph, start, goal, heuristic)` | A* with optional heuristic |
