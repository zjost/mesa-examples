"""
Configure visualization elements and instantiate a server
"""

from functools import partial

import mesa
import networkx as nx
import solara
from aco_tsp.model import EvoAntTspModel, TSPGraph
from matplotlib.figure import Figure
from mesa.visualization import SolaraViz


def circle_portrayal_example(agent):
    return {"node_size": 20, "width": 0.1}


tsp_graph = TSPGraph.from_tsp_file("aco_tsp/data/kroA100.tsp")
model_params = {
    "num_agents": tsp_graph.num_cities,
    "tsp_graph": tsp_graph,
    "num_winners": tsp_graph.num_cities // 5,
    "q_0_mean": {
        "type": "SliderFloat",
        "value": 0.9,
        "label": "q_0 mean: determinism threshold",
        "min": 0.0,
        "max": 1.0,
        "step": 0.1,
    },
    "beta_mean": {
        "type": "SliderFloat",
        "value": 2.0,
        "label": "Beta mean: heuristic exponent",
        "min": 0.0,
        "max": 10.0,
        "step": 0.1,
    },
    "noise_std": {
        "type": "SliderFloat",
        "value": 0.1,
        "label": "Mutation std",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
    },
    "ro": {
        "type": "SliderFloat",
        "value": 0.1,
        "label": "ro: 1-evaporation rate",
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
    if min_distance < opt_distance:
        path = model.best_path
        print(f"Distance: {min_distance}, Path: {path}")
    ax.set(
        title=f"Best distance = {min_distance - opt_distance:.2f}",
        xlabel="Steps",
        ylabel="Distance",
    )
    ax.plot(
        model.datacollector.get_model_vars_dataframe()["best_distance"] - 21282,
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
    weights = [model.tsp_graph.pheromone[u][v] for u, v in graph.edges()]
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


def make_histogram(model, param_name):
    fig = Figure()
    ax = fig.subplots()
    ax.set_title(f"{param_name} histogram")
    # Get last iteration's alpha values
    parm_values = [getattr(agent, param_name) for agent in model.schedule.agents]
    ax.hist(parm_values, bins=20, color="blue", alpha=0.7, rwidth=0.85)
    ax.set(ylim=(0, model.num_agents), xlabel=param_name)
    solara.FigureMatplotlib(fig)


def ant_level_distances(model):
    # ant_distances = model.datacollector.get_agent_vars_dataframe()
    # Plot so that the step index is the x-axis, there's a line for each agent,
    # and the y-axis is the distance traveled
    # ant_distances['tsp_distance'].unstack(level=1).plot(ax=ax)
    pass


@solara.component
def build_page():
    return SolaraViz(
        EvoAntTspModel,
        model_params,
        space_drawer=None,
        measures=[
            "best_distance_iter",
            best_distance_from_optimal,
            "q_0_mean_sample",
            "beta_mean_sample",
            partial(make_histogram, param_name="q_0"),
            partial(make_histogram, param_name="beta"),
            make_graph,
        ],
        agent_portrayal=circle_portrayal_example,
        play_interval=0,  # 1,
    )


page = build_page()


if __name__ == "__main__":
    model_params = {
        "num_agents": tsp_graph.num_cities,
        "tsp_graph": tsp_graph,
        "num_winners": tsp_graph.num_cities // 5,
        "q_0_mean": 0.9,
        "beta_mean": 2.0,
        "noise_std": [0.01],  # , 0.05, 1.0],
        "ro": 0.1,
    }
    results = mesa.batch_run(
        EvoAntTspModel,
        parameters=model_params,
        iterations=1,
        max_steps=20,
        number_processes=1,
        data_collection_period=1,
        display_progress=True,
    )
    # pd.DataFrame(results).to_csv("evo_results_00.csv")
