from aco_tsp.model import TSPGraph, compute_path_distance

tsp_graph = TSPGraph.from_tsp_file("aco_tsp/data/pcb442.tsp")

canonical_path = list(range(1, tsp_graph.num_cities + 1))

assert len(canonical_path) == tsp_graph.num_cities
distance = compute_path_distance(canonical_path, tsp_graph.g, nint_flag=False)
distance_nint = compute_path_distance(canonical_path, tsp_graph.g, nint_flag=True)
print(f"Canonical path distance: {distance:.3f}; nint: {distance_nint}")
assert distance_nint == 221_440
