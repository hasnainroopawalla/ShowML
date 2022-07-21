from graph import Graph
from ant import Ant
from antnet import runAntNet

# Sample graph with edge cost
G = Graph(0.9,0.1,0.2)

G.add_edge('A','B', 2)
G.add_edge('B','C', 2)
G.add_edge('A','H', 2)
G.add_edge('H','G', 2)
G.add_edge('C','F', 1)
G.add_edge('F','G', 1)
G.add_edge('G','F', 1)
G.add_edge('F','C', 1)
G.add_edge('C','D', 10)
G.add_edge('E','D', 2)
G.add_edge('G','E', 2)

Ant.graph = G

source = 'A'
destination = 'D'
iterations = 20
num_episodes = 5

print("Path taken by ants on each episode with their cost")
for episode in range(1, num_episodes+1):
    antnet_path = runAntNet(G, source, destination, 0.6, 0.3, 0.7) # Replace with -> antnet_path = antnet(G, source, destination)
    print(antnet_path)
    ant_net_cost = G.get_path_cost(antnet_path)
    print(ant_net_cost)
    print()
    