
class Edge:

    def __init__(self, vert1, vert2, **kwargs):
        self.vert1 = vert1
        self.vert2 = vert2
        self.data  = kwargs

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

