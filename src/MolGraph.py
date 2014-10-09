
class MolGraph:

    def __init__(self):
        self.verts = []
        self.edges = []

    def add_vert(self, vert):
        self.verts.append(vert)

    def add_edge(self, edge):
        if edge.get_verts()[0] not in self.verts or edge.get_verts()[1] not in self.verts:
            raise Exception("Edge refers to non-existent vertex.")

        self.edges.append(edge)

    def get_verts(self):
        return self.verts

    def get_edges(self):
        return self.edges

class Vertex:
    __slots__ = ['data', 'edges']
    def __init__(self, **kwargs):
        self.data  = kwargs         # Keep track of each atom's features.
        self.edges = ()

    def get_degree(self):
        return len(self.edges)

    def add_edge(self, edge):
        if edge in self.edges:
            raise Exception("Edge is already associated with vertex.  Self loops are not currently supported.")
        self.edges = self.edges + (edge, )

    def get_neighbors(self):
        return [e.other_vertex(self) for e in self.edges], self.edges

    def __getattr__(self, name):
        if self.data.has_key(name):
            return self.data[name]
        else:
            raise AttributeError("%r object has no attribute %r" % (self.__class__, name))

class Edge:
    __slots__ = ['vert1', 'vert2', 'data']
    def __init__(self, vert1, vert2, **kwargs):
        self.vert1 = vert1
        self.vert2 = vert2
        self.data  = kwargs    # Each edge's features.

        # Register with the vertices.
        self.vert1.add_edge(self)
        self.vert2.add_edge(self)

    def get_verts(self):
        return (self.vert1, self.vert2)

    def other_vertex(self, vert):
        if self.vert1 == vert:
            return self.vert2
        elif self.vert2 == vert:
            return self.vert1
        else:
            raise Exception("Edge does not connect this vertex.")

    def __getattr__(self, name):
        if self.data.has_key(name):
            return self.data[name]
        else:
            raise AttributeError("%r object has no attribute %r" % (self.__class__, name))
