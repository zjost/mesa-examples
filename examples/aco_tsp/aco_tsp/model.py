from dataclasses import dataclass
from functools import partial
from typing import List, Optional

import line_profiler
import mesa
import networkx as nx
import numpy as np


@dataclass
class NodeCoordinates:
    idx: int
    city: int
    x: float
    y: float

    @classmethod
    def from_line(cls, line: str, idx: int):
        city, x, y = line.split()
        return cls(idx, int(city), float(x), float(y))


class TSPGraph:
    def __init__(self, g: nx.Graph, pheromone_init: float = 1e-6):
        self.g = g
        self.pheromone_init = pheromone_init
        self._cities = None
        self._city2idx = None
        self._add_edge_properties()

    @property
    def pos(self):
        return {k: v["pos"] for k, v in dict(self.g.nodes.data()).items()}

    @property
    def cities(self):
        if self._cities is None:
            self._cities = list(self.g.nodes)
        return self._cities

    @property
    def num_cities(self):
        return len(self.cities)

    @property
    def visibility(self):
        return self._visibility

    @property
    def pheromone(self):
        return self._pheromone

    def update_pheromone(self, i, j, value):
        self._pheromone[i][j] = value
        self._pheromone[j][i] = value

    def _add_edge_properties(self):
        self._visibility = np.zeros((self.num_cities, self.num_cities))
        self._pheromone = (
            np.ones((self.num_cities, self.num_cities)) * self.pheromone_init
        )

        for u, v in self.g.edges():
            u_x, u_y = self.g.nodes[u]["pos"]
            v_x, v_y = self.g.nodes[v]["pos"]
            distance = ((u_x - v_x) ** 2 + (u_y - v_y) ** 2) ** 0.5
            self.g[u][v]["distance"] = distance
            self._visibility[u][v] = 1 / distance
            self._visibility[v][u] = 1 / distance
            # self.g[u][v]["visibility"] = 1 / self.g[u][v]["distance"]
            # self.g[u][v]["int_distance"] = int(self.g[u][v]["distance"] + 0.5)
            # self.g[u][v]["pheromone"] = self.pheromone_init

    @classmethod
    def from_random(cls, num_cities: int, seed: int = 0) -> "TSPGraph":
        g = nx.random_geometric_graph(num_cities, 2.0, seed=seed).to_directed()

        return cls(g)

    @classmethod
    def from_tsp_file(cls, file_path: str, pheromone_init: float = 1e-6) -> "TSPGraph":
        with open(file_path) as f:
            lines = f.readlines()
            # Skip lines until reach the text "NODE_COORD_SECTION"
            while lines.pop(0).strip() != "NODE_COORD_SECTION":
                pass

            g = nx.Graph()
            for idx, line in enumerate(lines):
                if line.strip() == "EOF":
                    break
                node_coordinate = NodeCoordinates.from_line(line, idx)

                g.add_node(
                    node_coordinate.idx,  # node id will be the idx
                    city_id=node_coordinate.city,  # this property will store the integer city id
                    pos=(node_coordinate.x, node_coordinate.y),
                )

        # Add edges between all nodes to make a complete graph
        for u in g.nodes():
            for v in g.nodes():
                if u == v:
                    continue
                g.add_edge(u, v)

        return cls(g, pheromone_init=pheromone_init)


def compute_path_distance(path: List, g: nx.Graph, nint_flag: bool = False):
    distance = 0
    # Add the first city to the end to complete the round trip
    round_trip = [*path, path[0]]
    for i in range(len(round_trip) - 1):
        city_start, city_end = round_trip[i], round_trip[i + 1]
        if nint_flag:  # nint(x) = int(x + 0.5)
            distance += int(g[city_start][city_end]["distance"] + 0.5)
        else:
            distance += g[city_start][city_end]["distance"]
    return distance


def three_opt(path, g):
    best_path = path
    best_distance = compute_path_distance(best_path, g)
    for i in range(len(path) - 1):
        k, el = path[i], path[i + 1]
        for j in range(i + 2, len(path) - 1):
            p, q = path[j], path[j + 1]
            for t in range(j + 2, len(path) - 1):
                r, s = path[t], path[t + 1]
                if (
                    g[k][q]["distance"] + g[p][s]["distance"] + g[r][el]["distance"]
                    < g[k][el]["distance"] + g[p][q]["distance"] + g[r][s]["distance"]
                ):
                    # new_path = [...k] + [q, ..., r] + [l, ..., p] + [s, ...]
                    new_path = (
                        path[: i + 1]
                        + path[j + 1 : t + 1]
                        + path[i + 1 : j + 1]
                        + path[t + 1 :]
                    )
                    assert len(new_path) == len(path)
                    new_distance = compute_path_distance(path=new_path, g=g)
                    if new_distance < best_distance:
                        # print(f"Found a better path with distance {new_distance}")
                        best_path = new_path
                        best_distance = new_distance

    return best_path


class AntTSP(mesa.Agent):
    """
    An agent
    """

    def __init__(
        self,
        unique_id: int,
        model: mesa.Model,
        alpha: float = 1.0,
        beta: float = 5.0,
        q_0: Optional[float] = None,
        aco_flag: bool = False,
        three_opt_iters: int = 0,
    ):
        """
        Customize the agent
        """
        self.unique_id = unique_id
        self._alpha = alpha
        self._beta = beta
        self._q_0 = q_0
        # Assert that either aco is False, or alpha == 1.0
        assert not aco_flag or alpha == 1.0, "alpha must be 1.0 for ACO"
        self.aco_flag = aco_flag
        self.three_opt_iters = three_opt_iters
        super().__init__(unique_id, model)
        self._cities_visited = []
        self._cities_not_visited = set(model.tsp_graph.cities)
        self._traveled_distance = 0
        self.tsp_solution = model.tsp_graph.cities
        self.tsp_distance = float("inf")

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta = value

    @property
    def q_0(self):
        return self._q_0

    @q_0.setter
    def q_0(self, value):
        self._q_0 = value

    @line_profiler.profile
    def decide_next_city(self):
        pheromone = self.model.tsp_graph.pheromone
        visibility = self.model.tsp_graph.visibility

        current_city = self.pos
        candidates = list(self._cities_not_visited)
        if len(candidates) == 0:
            return current_city

        results = (
            pheromone[current_city][candidates] ** self.alpha
            * visibility[current_city][candidates] ** self.beta
        )

        norm = results.sum()
        results /= norm

        if self.q_0 is not None and self.random.random() <= self.q_0:
            new_city = candidates[np.argmax(results)]
        else:
            new_city = self.model.random.choices(candidates, weights=results)[0]

        return new_city

    def local_update(self, g, current_city, new_city):
        ro = self.model.ro
        tau_0 = self.model.tsp_graph.pheromone_init

        old_pheromone = self.model.tsp_graph.pheromone[current_city][new_city]
        new_pheromone = (1 - ro) * old_pheromone + ro * tau_0

        self.model.tsp_graph.update_pheromone(current_city, new_city, new_pheromone)

    def init_agent(self):
        city = self.model.random.choice(self.model.tsp_graph.cities)
        if self.pos is not None:
            self.model.grid.move_agent(self, city)
        else:
            self.model.grid.place_agent(self, city)
        self._cities_visited = [city]
        self._cities_not_visited = set(self.model.tsp_graph.cities) - {city}
        self._traveled_distance = 0

    def step(self):
        """
        Modify this method to change what an individual agent will do during each step.
        Can include logic based on neighbors states.
        """
        g = self.model.grid.G
        for idx in range(self.model.num_cities - 1):
            # Pick a random city that isn't in the list of cities visited
            current_city = self.pos
            new_city = self.decide_next_city()
            self._cities_visited.append(new_city)
            self._cities_not_visited.remove(new_city)
            self.model.grid.move_agent(self, new_city)
            if self.aco_flag:
                self.local_update(g, current_city, new_city)

        self._traveled_distance = compute_path_distance(
            self._cities_visited, g, nint_flag=False
        )

        for _ in range(self.three_opt_iters):
            new_path = three_opt(self._cities_visited, g)
            new_distance = compute_path_distance(path=new_path, g=g, nint_flag=False)
            if new_distance < self._traveled_distance:
                self._cities_visited = new_path
                self._traveled_distance = new_distance
            else:
                break

        self.tsp_solution = self._cities_visited.copy()
        self.tsp_distance = self._traveled_distance
        self.init_agent()


def calculate_pheromone_delta(
    tsp_solution: List[int], tsp_distance: float, q: float = 100
):
    results = {}
    for idx, start_city in enumerate(tsp_solution[:-1]):
        end_city = tsp_solution[idx + 1]
        results[(start_city, end_city)] = q / tsp_distance

    return results


def extract_attr_fn(model, attr_name):
    return getattr(model, attr_name, None)


class AntSystemTspModel(mesa.Model):
    """
    The model class holds the model-level attributes, manages the agents, and generally handles
    the global level of our model.

    There is only one model-level parameter: how many agents the model contains. When a new model
    is started, we want it to populate itself with the given number of agents.

    The scheduler is a special model component which controls the order in which agents are activated.
    """

    def __init__(
        self,
        num_agents: int = 20,
        tsp_graph: TSPGraph = TSPGraph.from_random(20),
        max_steps: int = int(1e6),
        ro: float = 0.5,
        ant_alpha: float = 1.0,
        ant_beta: float = 5.0,
        ant_q_0: Optional[float] = None,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.tsp_graph = tsp_graph
        self.num_cities = tsp_graph.num_cities
        self.all_cities = set(range(self.num_cities))
        self.max_steps = max_steps
        self.ro = ro
        self.schedule = mesa.time.RandomActivation(self)
        self.grid = mesa.space.NetworkGrid(tsp_graph.g)

        self.initialize_agents(ant_alpha, ant_beta, ant_q_0)
        self.initialize_data_collection()
        # Re-initialize pheromone levels
        self.tsp_graph._add_edge_properties()
        self.running = True

    def initialize_agents(
        self, ant_alpha: float, ant_beta: float, ant_q_0: Optional[float]
    ) -> None:
        for i in range(self.num_agents):
            agent = AntTSP(
                unique_id=i,
                model=self,
                alpha=ant_alpha,
                beta=ant_beta,
                q_0=ant_q_0,
                aco_flag=False,
            )
            self.schedule.add(agent)
            agent.init_agent()

    def initialize_data_collection(self) -> None:
        self.num_steps = 0
        self.best_path = range(self.num_cities)
        self.best_distance = float("inf")
        self.best_distance_iter = float("inf")

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "num_steps": partial(extract_attr_fn, attr_name="num_steps"),
                "best_distance": partial(extract_attr_fn, attr_name="best_distance"),
                "best_distance_iter": partial(
                    extract_attr_fn, attr_name="best_distance_iter"
                ),
                "best_path": partial(extract_attr_fn, attr_name="best_path"),
            },
            agent_reporters={
                "tsp_distance": partial(extract_attr_fn, attr_name="tsp_distance"),
                "tsp_solution": partial(extract_attr_fn, attr_name="tsp_solution"),
            },
        )

    def update_pheromone(self, q: float = 100):
        # tau_ij(t+1) = (1-ro)*tau_ij(t) + delta_tau_ij(t)
        # delta_tau_ij(t) = sum_k^M {Q/L^k} * I[i,j \in T^k]
        delta_tau_ij = {}
        for k, agent in enumerate(self.schedule.agents):
            delta_tau_ij[k] = calculate_pheromone_delta(
                agent.tsp_solution, agent.tsp_distance, q
            )

        for i, j in self.grid.G.edges():
            # Evaporate
            tau_ij = (1 - self.ro) * self.tsp_graph.pheromone[i][j]
            # Add ant's contribution
            for k, delta_tau_ij_k in delta_tau_ij.items():
                tau_ij += delta_tau_ij_k.get((i, j), 0.0)

            self.tsp_graph.update_pheromone(i, j, tau_ij)

    def collect_data(self):
        # Check len of cities visited by an agent
        best_instance_iter = float("inf")
        for agent in self.schedule.agents:
            # Check for best path
            if agent.tsp_distance < self.best_distance:
                self.best_distance = agent.tsp_distance
                self.best_path = agent.tsp_solution

            if agent.tsp_distance < best_instance_iter:
                best_instance_iter = agent.tsp_distance

        self.best_distance_iter = best_instance_iter

    def step(self):
        """
        A model step. Used for collecting data and advancing the schedule
        """
        # self.datacollector.collect(self)
        self.schedule.step()
        self.num_steps += 1
        self.collect_data()
        self.datacollector.collect(self)
        self.update_pheromone()

        if self.num_steps >= self.max_steps:
            self.running = False


class ACOTspModel(AntSystemTspModel):
    def initialize_agents(
        self, ant_alpha: float, ant_beta: float, ant_q_0: Optional[float]
    ) -> None:
        for i in range(self.num_agents):
            agent = AntTSP(
                unique_id=i,
                model=self,
                alpha=1.0,
                beta=ant_beta,
                q_0=ant_q_0,
                aco_flag=True,
                three_opt_iters=0,
            )
            self.schedule.add(agent)
            agent.init_agent()

    def update_pheromone(self):
        # Global update of best path
        delta_tau_ij = calculate_pheromone_delta(
            self.best_path, self.best_distance, q=1.0
        )

        for i, j in delta_tau_ij:
            # Evaporate
            # tau_ij = (1 - self.ro) * self.grid.G[i][j]["pheromone"]
            tau_ij = (1 - self.ro) * self.tsp_graph.pheromone[i][j]
            # Add ant's contribution
            tau_ij += self.ro * delta_tau_ij[(i, j)]

            self.tsp_graph.update_pheromone(i, j, tau_ij)


class EvoAntTspModel(ACOTspModel):
    def __init__(
        self,
        num_agents: int = 20,
        num_winners: int = 5,
        tsp_graph: TSPGraph = TSPGraph.from_random(20),
        max_steps: int = int(1e6),
        ro: float = 0.1,
        q_0_mean: float = 0.9,
        beta_mean: float = 2.0,
        noise_std: float = 1.0,
    ):
        # Call super of base's base class
        mesa.Model.__init__(self)
        self.num_agents = num_agents
        self.num_winners = num_winners
        assert (
            num_agents % num_winners == 0
        ), "num_agents must be divisible by num_winners"
        self.offspring_per_winner = num_agents // num_winners

        self.tsp_graph = tsp_graph
        self.num_cities = tsp_graph.num_cities
        self.all_cities = set(range(self.num_cities))
        self.max_steps = max_steps
        self.ro = ro
        self.schedule = mesa.time.RandomActivation(self)
        self.grid = mesa.space.NetworkGrid(tsp_graph.g)

        self.q_0_mean = q_0_mean
        self.beta_mean = beta_mean
        self.noise_std = noise_std
        self.initialize_agents()
        self.initialize_data_collection()
        # Re-initialize pheromone levels
        self.tsp_graph._add_edge_properties()
        self.running = True

    def initialize_agents(self) -> None:
        for i in range(self.num_agents):
            q_0 = np.random.normal(loc=self.q_0_mean, scale=0.1)
            ant_beta = np.random.normal(loc=self.beta_mean, scale=0.1)
            agent = AntTSP(
                unique_id=i,
                model=self,
                alpha=1.0,
                beta=ant_beta,
                q_0=q_0,
                aco_flag=True,
            )
            self.schedule.add(agent)
            agent.init_agent()

    def initialize_data_collection(self) -> None:
        self.num_steps = 0
        self.best_path = None
        self.best_distance = float("inf")
        self.best_distance_iter = float("inf")

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "num_steps": partial(extract_attr_fn, attr_name="num_steps"),
                "best_distance": partial(extract_attr_fn, attr_name="best_distance"),
                "best_distance_iter": partial(
                    extract_attr_fn, attr_name="best_distance_iter"
                ),
                "best_path": partial(extract_attr_fn, attr_name="best_path"),
                "q_0_mean_sample": partial(
                    extract_attr_fn, attr_name="q_0_mean_sample"
                ),
                "beta_mean_sample": partial(
                    extract_attr_fn, attr_name="beta_mean_sample"
                ),
            },
            agent_reporters={
                "tsp_distance": partial(extract_attr_fn, attr_name="tsp_distance"),
                "tsp_solution": partial(extract_attr_fn, attr_name="tsp_solution"),
                "q_0": partial(extract_attr_fn, attr_name="q_0"),
                "beta": partial(extract_attr_fn, attr_name="beta"),
            },
        )

    def collect_data(self):
        super().collect_data()
        self.q_0_mean_sample = np.mean([agent.q_0 for agent in self.schedule.agents])
        self.beta_mean_sample = np.mean([agent.beta for agent in self.schedule.agents])

    def select_winners(self):
        # Sort agents by their performance and select the top num_winners
        winning_agents = sorted(self.schedule.agents, key=lambda x: x.tsp_distance)
        return winning_agents[: self.num_winners]

    def reproduce_and_mutate(self, winning_agents):
        # Create offspring for each winner
        for winner_idx, winner in enumerate(winning_agents):
            for off_idx in range(self.offspring_per_winner):
                # Mutate: if parent mean is mean_p, then child mean is mean_c = mean_p + N(0, 1)
                q_0 = np.clip(
                    np.random.normal(loc=winner.q_0, scale=self.noise_std), 0, 1
                )
                beta = np.random.normal(loc=winner.beta, scale=self.noise_std)
                agent_idx = winner_idx * self.offspring_per_winner + off_idx
                # Update agent params
                agent = self.schedule.agents[agent_idx]
                agent.q_0 = q_0
                agent.beta = beta

    def step(self):
        super().step()
        # Evolution
        winning_agents = self.select_winners()
        self.reproduce_and_mutate(winning_agents)
