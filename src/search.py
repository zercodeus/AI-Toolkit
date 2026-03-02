"""
search.py — Classic AI search algorithms: BFS, DFS, and A*.
"""

from collections import deque
import heapq


def bfs(graph: dict, start, goal) -> list:
    """
    Breadth-First Search — finds the shortest path (fewest edges).

    Args:
        graph: Adjacency dict {node: [neighbors]}.
        start: Starting node.
        goal:  Target node.

    Returns:
        List of nodes from start to goal, or empty list if no path found.
    """
    if start == goal:
        return [start]
    visited = {start}
    queue = deque([[start]])
    while queue:
        path = queue.popleft()
        node = path[-1]
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                new_path = path + [neighbor]
                if neighbor == goal:
                    return new_path
                visited.add(neighbor)
                queue.append(new_path)
    return []


def dfs(graph: dict, start, goal, visited=None) -> list:
    """
    Depth-First Search — finds a path (not necessarily shortest).

    Args:
        graph:   Adjacency dict {node: [neighbors]}.
        start:   Starting node.
        goal:    Target node.
        visited: Set of already-visited nodes (used internally).

    Returns:
        List of nodes from start to goal, or empty list if not found.
    """
    if visited is None:
        visited = set()
    visited.add(start)
    if start == goal:
        return [start]
    for neighbor in graph.get(start, []):
        if neighbor not in visited:
            path = dfs(graph, neighbor, goal, visited)
            if path:
                return [start] + path
    return []


def astar(graph: dict, start, goal, heuristic=None) -> list:
    """
    A* Search — finds the optimal path using a heuristic.

    Args:
        graph:     Weighted adjacency dict {node: [(neighbor, cost)]}.
        start:     Starting node.
        goal:      Target node.
        heuristic: Function h(node, goal) → estimated cost.
                   Defaults to zero heuristic (equivalent to Dijkstra).

    Returns:
        List of nodes from start to goal, or empty list if not found.
    """
    if heuristic is None:
        heuristic = lambda n, g: 0

    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start, [start]))
    visited = {}

    while open_set:
        f, g, node, path = heapq.heappop(open_set)
        if node == goal:
            return path
        if node in visited and visited[node] <= g:
            continue
        visited[node] = g
        for neighbor, cost in graph.get(node, []):
            new_g = g + cost
            new_f = new_g + heuristic(neighbor, goal)
            heapq.heappush(open_set, (new_f, new_g, neighbor, path + [neighbor]))

    return []
