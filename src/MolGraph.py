
class MolGraph(object):
    def __init__(self):
        self.verts = []
        self.edges = []

    def add_vert(self, vert):
        self.verts.append(vert)

    def add_edge(self, edge):
        assert edge.get_verts()[0] in self.verts and edge.get_verts()[1] in self.verts, "Edge refers to non-existent vertex."
        self.edges.append(edge)

    def get_verts(self):
        return self.verts

    def get_edges(self):
        return self.edges

class Vertex(object):
    def __init__(self):
        self.edges = ()

    def get_degree(self):
        return len(self.edges)

    def add_edge(self, edge):
        assert edge not in self.edges, "Edge is already associated with vertex.  Self loops are not currently supported."
        self.edges = self.edges + (edge, )

    def get_neighbors(self):
        return [e.other_vertex(self) for e in self.edges], self.edges

class Edge(object):
    def __init__(self, vert1, vert2):
        self.vert1 = vert1
        self.vert2 = vert2

        # Register with the vertices.
        self.vert1.add_edge(self)
        self.vert2.add_edge(self)

    def get_verts(self):
        return (self.vert1, self.vert2)

    def other_vertex(self, vert):
        if self.vert1 is vert:
            return self.vert2
        elif self.vert2 is vert:
            return self.vert1
        else:
            raise Exception("Edge does not connect this vertex.")

