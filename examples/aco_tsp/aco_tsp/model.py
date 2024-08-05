from typing import Dict, Tuple

import networkx as nx
import numpy as np

import mesa


def create_graph(num_cities: int, pheromone_init: float = 1e-6, seed: int = 0) -> Tuple[nx.Graph, Dict]:
    g = nx.random_geometric_graph(num_cities, 2., seed=seed).to_directed()
    pos = {k: v['pos'] for k, v in dict(g.nodes.data()).items()}

    for u, v in g.edges():
        u_x, u_y = g.nodes[u]['pos']
        v_x, v_y = g.nodes[v]['pos']
        g[u][v]['distance'] = ((u_x - v_x) ** 2 + (u_y - v_y) ** 2) ** 0.5
        g[u][v]['visibility'] = 1 / g[u][v]['distance']
        g[u][v]['pheromone'] = pheromone_init

    return g, pos


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

    def __init__(self, num_agents: int, num_cities: int, g: nx.Graph, pos: Dict):
        super().__init__()
        self.num_agents = num_agents
        self.num_cities = num_cities
        self.all_cities = set(range(self.num_cities))
        self.schedule = mesa.time.RandomActivation(self)
        self.pos = pos
        self.grid = mesa.space.NetworkGrid(g)

        for i in range(self.num_agents):
            agent = AntTSP(i, self)
            self.schedule.add(agent)

            city = self.random.randrange(self.num_cities)
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