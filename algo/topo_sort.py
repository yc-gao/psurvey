#!/usr/bin/env python3

from collections import defaultdict, deque


def topo_sort(tasks):
    # Create an adjacency list representation of the graph and in-degree count
    graph = defaultdict(list)
    in_degree = defaultdict(int)

    # Build the graph from the task list
    for task in tasks:
        node = task["node"]
        dependencies = task["deps"]

        # Ensure every node is accounted for in the in-degree count
        for dep in dependencies:
            graph[dep].append(node)
            in_degree[node] += 1

    # Queue for nodes with no incoming edges
    queue = deque([node for node in in_degree if in_degree[node] == 0])
    topo_order = []  # List to store the topological order
    # Process nodes with no incoming edges
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        # Decrease the in-degree of each neighbor, and if it becomes 0, add it to the queue
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    # Check if topo_order contains all nodes
    if len(topo_order) == len(tasks):
        return topo_order
    else:
        return []  # Return an empty list if there is a cycle and topo sort is not possible


# Example usage
tasks = [
    {"node": "A", "deps": [], "name": "123"},
    {"node": "B", "deps": ["A"]},
    {"node": "C", "deps": ["A"]},
    {"node": "D", "deps": ["B", "C"]}
]

topo_sort(tasks)

print(tasks)
