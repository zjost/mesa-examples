"""
Configure visualization elements and instantiate a server
"""

import networkx as nx
import solara
from aco_tsp.model import ACOTspModel, TSPGraph  # AntSystemTspModel
from matplotlib.figure import Figure
from mesa.visualization import SolaraViz


def circle_portrayal_example(agent):
    return {"node_size": 20, "width": 0.1}


# pheromone_init = 1/(n*Lnn)
l_nn = 24_225
pheromone_init = 1 / (100 * l_nn)
tsp_graph = TSPGraph.from_tsp_file(
    "aco_tsp/data/kroA100.tsp", pheromone_init=pheromone_init
)
model_params = {
    "num_agents": tsp_graph.num_cities,
    "tsp_graph": tsp_graph,
    "ant_alpha": {
        "type": "SliderFloat",
        "value": 1.0,
        "label": "Alpha: pheromone exponent",
        "min": 0.0,
        "max": 10.0,
        "step": 0.1,
    },
    "ant_beta": {
        "type": "SliderFloat",
        "value": 5.0,
        "label": "Beta: heuristic exponent",
        "min": 0.0,
        "max": 10.0,
        "step": 0.1,
    },
    "ant_q_0": {
        "type": "SliderFloat",
        "value": 0.9,
        "label": "q_0: probability of exploitation",
        "min": 0.0,
        "max": 1.0,
        "step": 0.1,
    },
}


def best_distance(model):
    fig = Figure()
    ax = fig.subplots()
    ax.set(title="Best distance", xlabel="Steps", ylabel="Distance")
    ax.plot(model.datacollector.get_model_vars_dataframe()["best_distance"])
    ax.hlines(21282, 0, model.num_steps, color="red", linestyle="--")
    solara.FigureMatplotlib(fig)


def best_distance_from_optimal(model):
    fig = Figure()
    ax = fig.subplots()
    min_distance = model.datacollector.get_model_vars_dataframe()["best_distance"].min()
    opt_distance = 21282
    ax.set(
        title=f"Best distance = {min_distance - opt_distance:.2f}",
        xlabel="Steps",
        ylabel="Distance",
    )
    ax.plot(
        model.datacollector.get_model_vars_dataframe()["best_distance"] - opt_distance,
        marker="o",
        ms=3,
    )
    ax.grid()
    solara.FigureMatplotlib(fig)


def make_graph(model):
    fig = Figure()
    ax = fig.subplots()
    ax.set_title("Cities and pheromone trails")
    graph = model.grid.G
    pos = model.tsp_graph.pos
    weights = [graph[u][v]["pheromone"] for u, v in graph.edges()]
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
    # AntSystemTspModel,
    ACOTspModel,
    model_params,
    space_drawer=None,
    measures=["best_distance_iter", best_distance_from_optimal, make_graph],
    agent_portrayal=circle_portrayal_example,
    play_interval=1,
)
