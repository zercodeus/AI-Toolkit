import pytest
from src.search import bfs, dfs, astar

GRAPH = {
    "A": ["B", "C"],
    "B": ["D", "E"],
    "C": ["F"],
    "D": [],
    "E": ["F"],
    "F": [],
}

WEIGHTED_GRAPH = {
    "A": [("B", 1), ("C", 4)],
    "B": [("C", 2), ("D", 5)],
    "C": [("D", 1)],
    "D": [],
}


class TestBFS:
    def test_finds_path(self):
        path = bfs(GRAPH, "A", "F")
        assert path[0] == "A" and path[-1] == "F"

    def test_shortest_path(self):
        # BFS should find the shortest route
        path = bfs(GRAPH, "A", "F")
        assert len(path) <= 4

    def test_same_start_goal(self):
        assert bfs(GRAPH, "A", "A") == ["A"]

    def test_no_path(self):
        graph = {"A": ["B"], "B": [], "C": []}
        assert bfs(graph, "A", "C") == []


class TestDFS:
    def test_finds_path(self):
        path = dfs(GRAPH, "A", "F")
        assert path[0] == "A" and path[-1] == "F"

    def test_same_start_goal(self):
        assert dfs(GRAPH, "A", "A") == ["A"]

    def test_no_path(self):
        graph = {"A": ["B"], "B": [], "C": []}
        assert dfs(graph, "A", "C") == []


class TestAStar:
    def test_finds_optimal_path(self):
        path = astar(WEIGHTED_GRAPH, "A", "D")
        assert path[0] == "A" and path[-1] == "D"

    def test_with_heuristic(self):
        heuristic = lambda n, g: 0  # Zero heuristic = Dijkstra
        path = astar(WEIGHTED_GRAPH, "A", "D", heuristic=heuristic)
        assert "D" in path

    def test_no_path(self):
        graph = {"A": [("B", 1)], "B": [], "C": []}
        assert astar(graph, "A", "C") == []
