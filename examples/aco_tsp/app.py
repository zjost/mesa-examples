"""
Configure visualization elements and instantiate a server
"""
import solara
from matplotlib.figure import Figure
import networkx as nx

from mesa.visualization import SolaraViz

from aco_tsp.model import AcoTspModel, TSPGraph  # noqa


def circle_portrayal_example(agent):
    return {"node_size": 20, "width": 0.1}

tsp_graph = TSPGraph.from_tsp_file("aco_tsp/data/kroA100.tsp")
model_params = {
    "num_agents": tsp_graph.num_cities, 
    "tsp_graph": tsp_graph,
    "ant_alpha": {
        "type": "SliderFloat",
        "value": 1.,
        "label": "Alpha: pheromone exponent",
        "min": 0.,
        "max": 10.,
        "step": 0.1,
    },
    "ant_beta": {
        "type": "SliderFloat",
        "value": 5.,
        "label": "Beta: heuristic exponent",
        "min": 0.,
        "max": 10.,
        "step": 0.1,
    },
}

def make_graph(model):
    fig = Figure()
    ax = fig.subplots()
    graph = model.grid.G
    pos = model.tsp_graph.pos
    weights = [graph[u][v]['pheromone'] for u, v in graph.edges()]
    # normalize the weights
    weights = [w / max(weights) for w in weights]

    nx.draw(
        graph,
        ax=ax,
        pos=pos,
        node_size=10,
        width=weights,
        edge_color="gray",
    )
    
    solara.FigureMatplotlib(fig)

def ant_level_distances(model):
    # ant_distances = model.datacollector.get_agent_vars_dataframe()
    # Plot so that the step index is the x-axis, there's a line for each agent, 
    # and the y-axis is the distance traveled
    # ant_distances['tsp_distance'].unstack(level=1).plot(ax=ax)
    pass

page = SolaraViz(
    AcoTspModel,
    model_params,
    measures=["best_distance_iter", "best_distance", make_graph],
    agent_portrayal=circle_portrayal_example,
    play_interval=1,
)
