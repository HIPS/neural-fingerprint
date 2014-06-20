
class Edge:

    def __init__(self, **kwargs):
        self.data = kwargs

    def __getattr__(self, name):
        if self.data.has_key(name):
            return self.data[name]
        else:
            raise AttributeError("%r object has no attribute %r" % (self.__class__, name))
