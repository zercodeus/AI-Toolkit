import pytest
from src.nlp import tokenize, remove_stopwords, term_frequency, cosine_similarity, tfidf
from src.preprocessing import normalize, standardize, train_test_split, one_hot_encode, flatten


# ─── NLP Tests ───────────────────────────────────────────────────────────────

class TestTokenize:
    def test_basic(self):
        assert tokenize("Hello world") == ["hello", "world"]

    def test_punctuation_removed(self):
        assert tokenize("AI is great!") == ["ai", "is", "great"]

    def test_lowercase(self):
        assert tokenize("Artificial Intelligence") == ["artificial", "intelligence"]

    def test_invalid_input(self):
        with pytest.raises(TypeError):
            tokenize(123)


class TestRemoveStopwords:
    def test_removes_common_words(self):
        tokens = ["the", "cat", "is", "on", "table"]
        result = remove_stopwords(tokens)
        assert "the" not in result
        assert "cat" in result

    def test_custom_stopwords(self):
        result = remove_stopwords(["good", "bad"], stopwords={"bad"})
        assert result == ["good"]


class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert cosine_similarity([1, 0, 1], [1, 0, 1]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_zero_vector(self):
        assert cosine_similarity([0, 0], [1, 1]) == 0.0

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            cosine_similarity([1, 2], [1])


class TestTFIDF:
    def test_returns_dict(self):
        corpus = [["ai", "machine", "learning"], ["deep", "learning", "ai"]]
        result = tfidf(["ai", "learning"], corpus)
        assert isinstance(result, dict)

    def test_common_term_lower_score(self):
        corpus = [["ai", "learning"], ["ai", "deep"], ["ai", "nlp"]]
        scores = tfidf(["ai", "learning"], corpus)
        # "ai" appears in all docs → lower IDF than "learning"
        assert scores.get("ai", 0) <= scores.get("learning", 0)


# ─── Preprocessing Tests ──────────────────────────────────────────────────────

class TestNormalize:
    def test_range(self):
        result = normalize([0, 5, 10])
        assert result[0] == 0.0 and result[-1] == 1.0

    def test_empty(self):
        assert normalize([]) == []

    def test_all_same(self):
        assert normalize([3, 3, 3]) == [0.0, 0.0, 0.0]


class TestStandardize:
    def test_mean_zero(self):
        result = standardize([2, 4, 6])
        assert sum(result) == pytest.approx(0.0, abs=1e-9)

    def test_empty(self):
        assert standardize([]) == []


class TestTrainTestSplit:
    def test_sizes(self):
        train, test = train_test_split(list(range(100)), test_size=0.2, seed=42)
        assert len(train) == 80
        assert len(test) == 20

    def test_no_overlap(self):
        data = list(range(50))
        train, test = train_test_split(data, test_size=0.2, shuffle=False)
        assert set(train).isdisjoint(set(test))

    def test_invalid_test_size(self):
        with pytest.raises(ValueError):
            train_test_split([1, 2, 3], test_size=1.5)


class TestOneHotEncode:
    def test_basic(self):
        encoded, classes = one_hot_encode(["cat", "dog", "cat"])
        assert classes == ["cat", "dog"]
        assert encoded[0] == [1, 0]
        assert encoded[1] == [0, 1]

    def test_single_class(self):
        encoded, classes = one_hot_encode(["a", "a"])
        assert classes == ["a"]
        assert all(v == [1] for v in encoded)


class TestFlatten:
    def test_nested(self):
        assert flatten([[1, 2], [3, [4, 5]]]) == [1, 2, 3, 4, 5]

    def test_already_flat(self):
        assert flatten([1, 2, 3]) == [1, 2, 3]
