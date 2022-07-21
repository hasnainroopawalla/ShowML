import random
from pprint import pprint
import math

class Graph:

    def __init__(self, alpha=0.9, beta=0.1, evap = 0.1):
        """
        graph:-
        {
            'A':{
                'visited': False,
                'neighbors':{
                    'B': 3,
                    'C': 9
                },
                'pheros': {
                    'B': 5
                },
                'routing_table': {
                    'B': 0.2,
                    'C': 0.1
                },
                'traffic_stat': {
                    'B': {
                        'mean': 0.4,
                        'std': 0.2,
                        'W': [3, 4, 5]
                    },
                    'C': {
                        'mean': 0.4,
                        'std': 0.2,
                        'W': [3, 4, 5]
                    }
                }
            }
        }
        """

        self.graph = {}
        self.alpha = alpha
        self.beta = beta
        self.evap = evap
        self.w_max = 7


    def set_antnet_hyperparams(self, c1, c2, gamma):
        self.c1 = c1
        self.c2 = c2
        self.gamma = gamma


    def node_exists(self, node):
        return True if node in self.graph else False


    def edge_exists(self, source, destination):
        if not self.node_exists(source) or not self.node_exists(destination):
            return False
        return True if destination in self.graph[source]['neighbors'].keys() else False


    def add_node(self, node):
        self.graph[node] = {'neighbors':{},'routing_table':{node: 0.0}, 'pheromones':{}} #Set of pheromones with the neighbors


    """
    Check necessary nodes (source and destination) if not exist and create them.
    Add an edge with the travel time.
    """
    def add_edge(self, source, destination, travel_time):
        if not self.node_exists(source):
            self.add_node(source)
        if not self.node_exists(destination):
            self.add_node(destination)
        self.graph[source]['neighbors'][destination] = travel_time
        self.graph[source]['pheromones'][destination] = 1.0 #For every new edge, the pheromone value is initialized to 1
        self.graph[source]['routing_table'][destination] = 0.0
        self.graph[source]['traffic_stat'] = {}
        self.graph[source]['visited'] = False
        self.graph[destination]['visited'] = False


    def get_all_nodes(self):
        return list(self.graph.keys())


    def get_all_edges(self):
        edges = []
        for source in self.graph:
            for destination in self.graph[source]['neighbors']:
                edges.append((source, destination, self.graph[source]['neighbors'][destination]))
        return edges


    def get_node(self, node):
        if self.node_exists(node):
            return self.graph[node]
        return None


    def get_neighbors(self, node):
        neighbors = []
        if self.node_exists(node):
            for neighbor in self.graph[node]['neighbors']:
                neighbors.append(neighbor)
        return neighbors


    # Getting the pheromones vlaues of the edges from which node is the source
    def get_pheromones(self, node):
        if self.node_exists(node):
            return list(self.graph[node]['pheromones'].values())
        return None 
    

    def get_travel_times(self, node):
        if self.node_exists(node):
            return list(self.graph[node]['neighbors'].values())
        return None
    

    def get_alpha(self):
        return self.alpha
    

    def get_beta(self):
        return self.beta
    
    def get_evaporation(self):
        return self.evap


    def get_edge_time(self, source, destination):
        if not self.node_exists(source):
            return None
        if not self.node_exists(destination):
            return None
        
        if destination in self.graph[source]['neighbors']:
            return self.graph[source]['neighbors'][destination]

        return None

    def delete_node(self, node):
        if self.node_exists(node):
            for n in self.graph:
                if node in self.graph[n]['neighbors']:
                    del self.graph[n]['neighbors'][node]
            del self.graph[node]


    def delete_edge(self, source, destination):
        if self.edge_exists(source, destination):
            del self.graph[source]['neighbors'][destination]


    def update_travel_time(self, source, destination, new_travel_time):
        if self.edge_exists(source, destination):
            if new_travel_time <= 0:
                new_travel_time = 1
            self.graph[source]['neighbors'][destination] = new_travel_time


    def get_path_cost(self, path):
        cost = 0
        for i in range(len(path)-1):
            if self.edge_exists(path[i],path[i+1]):
                cost += self.get_edge_time(path[i],path[i+1])   
            else:
                return float('inf')
        return cost


    def update_pheromones(self, source, destination):
        if self.edge_exists(source, destination):
            self.graph[source]['pheromones'][destination] += 1 


    def display_graph(self):
        for node in self.graph:
            print("--> NODE {} <--".format(node))
            for neighbor in self.graph[node]['neighbors']:
                cost = self.graph[node]['neighbors'][neighbor]
                pheros = self.graph[node]['pheromones'][neighbor]
                print("{} -> {} Cost: {}, Pheros: {}".format(node, neighbor, cost, pheros))

            print("Traffic Statistics to reach Destination")
            if 'traffic_stat' in self.graph[node]:
                for dest in self.graph[node]['traffic_stat']:
                    W = self.graph[node]['traffic_stat'][dest]['W']
                    mean = self.graph[node]['traffic_stat'][dest]['mean']
                    var = self.graph[node]['traffic_stat'][dest]['var']
                    print("|- W = {}".format(W))
                    print("|- Mean = {}".format(mean))
                    print("|- Variance = {}".format(var))
                    print()

            print("Routing Table Information")
            for dest in self.graph[node]['routing_table']:
                print(" |- Prob. to reach {} = {}".format(dest, self.graph[node]['routing_table'][dest]))
                
            print("-"*50)
            print("-"*50)

    def evaporate(self):
        for node in self.graph:
            for neighbor in self.graph[node]['pheromones']:
                self.graph[node]['pheromones'][neighbor] = (1-self.evap)*self.graph[node]['pheromones'][neighbor]


    def add_phero(self, source, destination):
        self.graph[source]['pheromones'][destination] += 1.0/self.graph[source]['neighbors'][destination]


    def update_graph(self, max_delta_time=2, update_probability=0.7): 
        '''
            max_delta_time: maximum allowed change in travel time of an edge (in positive or negative direction)
            update_probability: probability that the travel time of an edge will change
        ''' 
        for edge in self.get_all_edges():
            if random.random() <= update_probability: # update the edge
                delta_time = random.choice([i for i in range(-max_delta_time,max_delta_time+1,1) if i!=0]) # Change the travel time by delta_time units
                self.update_travel_time(edge[0], edge[1], edge[2]+delta_time)

    # updates the traffic_stat datastructure of the node
    def update_traffic_stat(self, node, destination, neighbor, t):
        # Update traffic status
        if destination in self.graph[node]['traffic_stat']:
            self.graph[node]['traffic_stat'][destination]['W'].append(t)
            self.graph[node]['traffic_stat'][destination]['mean'] = sum(self.graph[node]['traffic_stat'][destination]['W']) / len(self.graph[node]['traffic_stat'][destination]['W'])
            self.graph[node]['traffic_stat'][destination]['var'] = ((t - self.graph[node]['traffic_stat'][destination]['mean'])**2) / len(self.graph[node]['traffic_stat'][destination]['W'])
        else:
            self.graph[node]['traffic_stat'][destination] = {
                'W': [t],
                'mean': t,
                'var': 0
            }

        if len(self.graph[node]['traffic_stat'][destination]['W']) > self.w_max:
            self.graph[node]['traffic_stat'][destination]['W'].pop(0)

        # Update routing table
        t_best = min(self.graph[node]['traffic_stat'][destination]['W'])
        first_term = self.c1 * (t_best / t)

        try:
            conf = math.sqrt(1 - self.gamma)
            W_max = len(self.graph[node]['traffic_stat'][destination]['W'])
            t_sup = self.graph[node]['traffic_stat'][destination]['mean'] + (self.graph[node]['traffic_stat'][destination]['var'] / (conf * math.sqrt(W_max)))
            second_term = self.c2 * ((t_sup - t_best) / ((t_sup - t_best) + (t - t_best)))
        except ZeroDivisionError as e:
            second_term = 0

        r = first_term + second_term
        
        # print("r for {} with neighbor {} -> {}".format(node, neighbor, r))
        # print(first_term, second_term)

        self.graph[node]['routing_table'][neighbor] += r * (1 - self.graph[node]['routing_table'][neighbor])
        for n in self.graph[node]['routing_table']:
            if n == neighbor:
                continue
            self.graph[node]['routing_table'][n] -= r * self.graph[node]['routing_table'][n]
        

    def set_window_size(self, w_max):
        self.w_max = w_max