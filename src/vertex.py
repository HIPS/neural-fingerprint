
class Vertex:

    def __init__(self, **kwargs):
        self.data  = kwargs
        self.edges = []

    def get_degree(self):
        return len(self.edges)

    def add_edge(self, edge):
        if edge in self.edges:
            raise Exception("Edge is already associated with vertex.  Self loops are not currently supported.")
        self.edges.append(edge)

    def get_neighbors(self):
        return [e.other_vertex(self) for e in self.edges], self.edges

    def __getattr__(self, name):
        if self.data.has_key(name):
            return self.data[name]
        else:
            raise AttributeError("%r object has no attribute %r" % (self.__class__, name))

