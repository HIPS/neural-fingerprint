from edge   import Edge
from vertex import Vertex

class Graph:

    def __init__(self):
        self.vertices = {}
        self.edges    = {}

    def add_vertex(self, vertex):
        self.vertices[vertex] = {}

    def add_edge(self, vertex1, vertex2, edge):
        self.edges[edge] = {}

    def get_vertices(self):
        return self.vertices.keys()
