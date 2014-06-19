from edge   import Edge
from vertex import Vertex

class Graph:

    def __init__(self):
        self.vertices = {}
        self.edges    = {}

    def add_vertex(self, vertex):
        self.vertices[vertex] = {}

    
