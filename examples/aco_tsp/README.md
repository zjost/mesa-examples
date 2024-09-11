Ant System for the Traveling Salesman Problem
========================

This is an implementation of the Ant System (AS) and Ant Colony Optimization (ACO) algorithms for solving the Traveling Salesman Problem (TSP).  This example uses Mesa's Network Grid to model a TSP by representing cities as nodes and the possible paths between them as edges.  Ants are then modeled as Mesa Agents that generate solutions by traversing the network using "swarm intelligence" algorithms.

When an ant is choosing its next city, it considers both a pheromone trail laid down by previous ants and a greedy heuristic based on city proximity.  Pheromone evaporates over time and the strength of new pheromone trail laid by an ant is proportional to the quality of its TSP solution.  This produces an emergent solution as the pheromone trail is continually updated and guides ants to high quality solutions as they are discovered.

As this model runs, more pheromone will be laid on better solutions, and less desirable paths will have their pheromone evaporate.  Ants will therefore reinforce good paths and abandon bad ones.  Since decisions are ultimately samples from a weighted probability distribution, ants will sometimes explore unlikely paths, which might lead to new strong solutions that will be reflected in the updated pheromone levels.

Here, we plot the best solution per iteration, the best solution so far in all iterations, and a graph representation where the edge width is proportional to the pheromone quantity.

## How to run
To launch the interactive visualization, run `solara run all_apps.py` in this directory.  Tune the parameters to influence the ants' decisions.  See the Algorithm details section for more.

## Algorithm details
### Ant System
Ant System (AS) was the first iteration of this family of algorithm.  Each agent/ant is initialized to a random city and constructs a solution by choosing a sequence of cities until all are visited, but none are visited more than once.  Ants then deposit a "pheromone" signal on each path in their solution that is proportional to 1/d, where d is the final distance of the solution.  This means shorter paths are given more pheromone.

When an ant is on city $i$ and deciding which city to choose next, it samples randomly using the following probabilities of transition from city $i$ to $j$:

$$
p_{ij}^k =  \frac{\tau_{ij}^\alpha \eta_{ij}^\beta}{\sum_{l \in J_i^k} \tau_{il}^\alpha \eta_{il}^\beta}
$$

where:

- $\tau_{ij}$ is the amount of path pheromone
- $\eta_{ij}$ the a greedy heuristic of desireability
  - In this case, $\eta_{ij} = 1/d_{ij}$, where $d_{ij}$ is the distance between
    cities
- $\alpha$ is a hyperparameter setting the importance of the pheromone
- $\beta$ a hyperparameter for setting the importance of the greedy heuristic
- And the denominator sum is over $J_i^k$, which is the set of cities not yet
  visited by ant $k$.

In other words, $\alpha$ and $\beta$ are tuned to set the relative importance of the phermone trail left by prior ants, and the greedy heuristic of 1-over-distance.

### Ant Colony Optimization
Ant Colony Optimization (ACO) is similar to AS, but received a number of tweaks to address failure modes of AS.  The primary difference is that instead of having all ants lay new pheromone, only the absolute best path is re-inforced.  This is known as the "global update".  Additionally, instead of evaporating pheromone on every path, only the paths that ants traverse have evaporation.  This is known as the "local" update since it is applied as ants take individual steps, not at the end of solution construction.  The net effect of the local update process is that subsequent ants are encouraged *not* to follow paths that previous ants explored, which increases the diversity of solutions that are explored.

Another difference is the transition probability function.  First, the $\alpha$ parameter is simply set to 1.0.  Next, a $q_0$ parameter is introduced.  When determining the next city, a uniform random number is drawn and if it's below $q_0$, the ant deterministically chooses the city with the highest probability (i.e., "exploit").  If the random number is greater than $q_0$, then the Ant System sampling process described above is used (i.e., "explore").  This allows you to explicitly control the explore vs exploit dynamic.

Qualitatively, you'll notice that ACO tends to leave more explore options available than AS, which quickly evaporates initially unlikely solutions and gets trapped in paths that receive early reinforcement.

## Data collection
The following data is collected and can be used for further analysis:
- Agent-level (individual ants, reset after each iteration)
  - `tsp_distance`: TSP solution distance
  - `tsp_solution`: TSP solution path
- Model-level (collection of ants over many iterations)
  - `num_steps`: number of algorithm iterations, where one step means each ant generates a full TSP solution and the pheromone trail is updated
  - `best_distance`: the distance of the best path found in all iterations
    - This is the best solution yet and can only stay flat or improve over time
  - `best_distance_iter`: the distance of the best path of all ants in a single iteration
    - This changes over time as the ant colony explores different solutions and can be used to understand the explore/exploit trade-off.  E.g., if the colony quickly finds a good solution, but then this value trends upward and stays high, then this suggests the ants are stuck re-inforcing a suboptimal solution.
  - `best_path`: the best path found in all iterations

## References
- Original paper:  Dorigo, M., Maniezzo, V., & Colorni, A. (1996). Ant system: optimization by a colony of cooperating agents. IEEE transactions on systems, man, and cybernetics, part b (cybernetics), 26(1), 29-41.
- ACO paper:  Dorigo, M., & Gambardella, L.M. (1997). Ant colony system: a cooperative learning approach to the traveling salesman problem. IEEE Trans. Evol. Comput., 1, 53-66.
- Video series of this code being implemented:  https://www.youtube.com/playlist?list=PLSgGvve8UweGk2TLSO-q5OSH59Q00ZxCQ