from dataclasses import dataclass
from typing import Dict, Tuple


import networkx as nx
import numpy as np

import mesa


@dataclass
class NodeCoordinates:
    city: int
    x: float
    y: float

    @classmethod
    def from_line(cls, line: str):
        city, x, y = line.split()
        return cls(int(city), float(x), float(y))
    
class TSPGraph:

    def __init__(self, g: nx.Graph, pheromone_init: float = 1e-6):
        self.g = g
        self.pheromone_init = pheromone_init
        self._add_edge_properties()

    def update_pheromone(
        self, 
        model: mesa.Model, 
        q: float = 100, 
        ro: float=0.5
    ):
        # tau_ij(t+1) = (1-ro)*tau_ij(t) + delta_tau_ij(t)
        # delta_tau_ij(t) = sum_k^M {Q/L^k} * I[i,j \in T^k]
        delta_tau_ij = dict()
        for k, agent in enumerate(model.schedule.agents):
            delta_tau_ij[k] = agent.calculate_pheromone_delta(q)

        for i, j in self.g.edges():
            # Evaporate
            tau_ij =  (1-ro)*self.g[i][j]['pheromone']
            # Add ant's contribution
            for k, delta_tau_ij_k in delta_tau_ij.items():
                tau_ij += delta_tau_ij_k.get((i,j), 0.)

            self.g[i][j]['pheromone'] = tau_ij

    @property
    def pos(self):
        return {k: v['pos'] for k, v in dict(self.g.nodes.data()).items()}
    
    @property
    def cities(self):
        return list(self.g.nodes)
    
    @property
    def num_cities(self):
        return len(self.g.nodes)
    
    def _add_edge_properties(self):
        for u, v in self.g.edges():
            u_x, u_y = self.g.nodes[u]['pos']
            v_x, v_y = self.g.nodes[v]['pos']
            self.g[u][v]['distance'] = ((u_x - v_x) ** 2 + (u_y - v_y) ** 2) ** 0.5
            self.g[u][v]['visibility'] = 1 / self.g[u][v]['distance']
            self.g[u][v]['pheromone'] = self.pheromone_init

    @classmethod
    def from_random(
        cls, 
        num_cities: int,  
        seed: int = 0
    ) -> 'TSPGraph':
        g = nx.random_geometric_graph(num_cities, 2., seed=seed).to_directed()

        return cls(g)
    
    @classmethod
    def from_tsp_file(cls, file_path: str) -> 'TSPGraph':
        with open(file_path, "r") as f:
            lines = f.readlines()
            # Skip lines until reach the text "NODE_COORD_SECTION"
            while lines.pop(0).strip() != "NODE_COORD_SECTION":
                pass

            g = nx.Graph()
            for line in lines:
                if line.strip() == "EOF":
                    break
                node_coordinate = NodeCoordinates.from_line(line)

                g.add_node(
                    node_coordinate.city, 
                    pos=(node_coordinate.x, node_coordinate.y)
                )

        # Add edges between all nodes to make a complete graph
        for u in g.nodes():
            for v in g.nodes():
                if u == v:
                    continue
                g.add_edge(u, v)
            
        return cls(g)


class AntTSP(mesa.Agent):  # noqa
    """
    An agent
    """

    def __init__(self, unique_id, model):
        """
        Customize the agent
        """
        self.unique_id = unique_id
        super().__init__(unique_id, model)
        self.cities_visited = list()
        self.traveled_distance = 0

    def calculate_pheromone_delta(self, q: float = 100):
        results = dict()
        for idx, start_city in enumerate(self.cities_visited[:-1]):
            end_city = self.cities_visited[idx+1]
            results[(start_city, end_city)] = q/self.traveled_distance

        return results

    
    def decide_next_city(self, alpha: float = 1.0, beta: float = 5.0):
        # Random
        # new_city = self.random.choice(list(self.model.all_cities - set(self.cities_visited)))
        # Choose closest city not yet visited
        g = self.model.grid.G
        current_city = self.pos
        neighbors = list(g.neighbors(current_city))
        candidates = [n for n in neighbors if n not in self.cities_visited]
        if len(candidates) == 0:
            return current_city


        # p_ij(t) = 1/Z*[(tau_ij)**alpha * (1/distance)**beta]
        results = list()
        for city in candidates:
            val = (g[current_city][city]["pheromone"])**alpha * (g[current_city][city]["visibility"])**beta
            results.append(val)
        
        results = np.array(results)
        Z = results.sum()
        results /= Z

        new_city = self.model.random.choices(candidates, weights=results)[0]
        self.traveled_distance += g[current_city][new_city]["distance"]
        return new_city

    def step(self):
        """
        Modify this method to change what an individual agent will do during each step.
        Can include logic based on neighbors states.
        """
        # Pick a random city that isn't in the list of cities visited
        new_city = self.decide_next_city()
        # print(f"Moving Ant {self.unique_id} from city {self.pos} to {new_city}")
        self.cities_visited.append(new_city)
        self.model.grid.move_agent(self, new_city)
        

class AcoTspModel(mesa.Model):
    """
    The model class holds the model-level attributes, manages the agents, and generally handles
    the global level of our model.

    There is only one model-level parameter: how many agents the model contains. When a new model
    is started, we want it to populate itself with the given number of agents.

    The scheduler is a special model component which controls the order in which agents are activated.
    """

    def __init__(self, num_agents: int, tsp_graph: TSPGraph):
        super().__init__()
        self.num_agents = num_agents
        self.num_cities = tsp_graph.num_cities
        self.all_cities = set(range(self.num_cities))
        self.schedule = mesa.time.RandomActivation(self)
        self.grid = mesa.space.NetworkGrid(tsp_graph.g)

        for i in range(self.num_agents):
            agent = AntTSP(i, self)
            self.schedule.add(agent)

            city = tsp_graph.cities[self.random.randrange(self.num_cities)]
            self.grid.place_agent(agent, city)
            agent.cities_visited.append(city) 

        self.num_steps = 0
        self.best_path = None
        self.best_distance = float('inf')

        self.datacollector = mesa.datacollection.DataCollector(
            model_reporters={"num_steps": "num_steps", "best_distance": "best_distance", "best_path": "best_path"},
            agent_reporters={"traveled_distance": "traveled_distance", "cities_visited": "cities_visited"}
        )

        self.running = True
        self.datacollector.collect(self)

    
    def step(self):
        """
        A model step. Used for collecting data and advancing the schedule
        """
        self.datacollector.collect(self)
        self.schedule.step()
        self.num_steps += 1

        # Check len of cities visited by an agent
        for agent in self.schedule.agents:
            if len(agent.cities_visited) == self.num_cities: 
                # Check for best path
                if agent.traveled_distance < self.best_distance:
                    self.best_distance = agent.traveled_distance
                    self.best_path = agent.cities_visited
                    # print(f"New best path found:  distance={self.best_distance}; path={self.best_path}")
                
                self.running = False