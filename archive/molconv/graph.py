import copy
import numpy        as np
import numpy.random as npr

from edge   import Edge
from vertex import Vertex

class Graph:

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

class GraphNet:

    def __init__(self, layers, max_degree):
        # Layers is a list of tuples with (vert_sz, edge_sz).
        self.layers     = layers
        self.max_degree = max_degree
        self.vert_wts   = []
        self.edge_wts   = []
        self.net_wts    = []
        self.net_bias   = 0.0

        self.init_weights()

    def init_weights(self, scale=0.01):

        # Convolutional weights.
        for layer in xrange(len(self.layers)-1):
            layer_vert_wts = [None]

            # Edge weights don't depend on degree.
            self.edge_wts.append({ 'v2e': scale*npr.randn( self.layers[layer+1][1],
                                                           self.layers[layer][0],
                                                           2),
                                   'e2e': scale*npr.randn( self.layers[layer+1][1],
                                                           self.layers[layer][1] ),
                                   'b2e': np.ones((self.layers[layer+1][1])) })
            
            # Vertex weights do depend on degree.
            for deg in xrange(1,self.max_degree+1):
                
                layer_vert_wts.append({ 'v2v': scale*npr.randn( self.layers[layer+1][0],
                                                                self.layers[layer][0],
                                                                deg+1 ),
                                        'e2v': scale*npr.randn( self.layers[layer+1][0],
                                                                self.layers[layer][1],
                                                                deg ),
                                        'b2v': np.ones((self.layers[layer+1][0])) })
            self.vert_wts.append(layer_vert_wts)

        # Top-level network weights. Just linear for now.
        self.net_wts = scale*npr.randn( self.layers[-1][0]+self.layers[-1][1] )

    def apply_conv(self, graph): # TODO: layer-specific dropout, variable nonlinearity.

        for ii in xrange(len(self.layers)-1):

            for vert in graph.get_verts():
                
                deg = vert.get_degree()
                
                # Start with biases
                vert.units[ii+1] = self.vert_wts[ii][deg]['b2v'].copy()

                # Get neighboring edges and vertices.
                n_verts, n_edges = vert.get_neighbors()

                # Loop over vertex neighbors, and self in previous layer.
                for jj, n_vert in enumerate([vert] + n_verts):

                    vert.units[ii+1] += np.dot( self.vert_wts[ii][deg]['v2v'][:,:,jj],
                                                n_vert.units[ii])

                # Loop over edge neighbors.
                for jj, n_edge in enumerate(n_edges):
                    vert.units[ii+1] += np.dot( self.vert_wts[ii][deg]['e2v'][:,:,jj],
                                                n_edge.units[ii])
                
                # Apply nonlinearity. TODO: in-place
                vert.units[ii+1] = np.maximum(vert.units[ii+1], 0.0)

            for edge in graph.get_edges():

                # Start with biases
                edge.units[ii+1] = self.edge_wts[ii]['b2e'].copy()

                # Update from self in previous layer.
                edge.units[ii+1] += np.dot( self.edge_wts[ii]['e2e'], edge.units[ii])

                # Only two neighboring vertices.
                n_vert1, n_vert2 = edge.get_verts()
                edge.units[ii+1] += np.dot( self.edge_wts[ii]['v2e'][:,:,0], n_vert1.units[ii] )
                edge.units[ii+1] += np.dot( self.edge_wts[ii]['v2e'][:,:,1], n_vert2.units[ii] )

                # Apply nonlinearity. TODO: in-place
                edge.units[ii+1] = np.maximum(edge.units[ii+1], 0.0)
        
        # TODO: make this more general.
        # For now, perform max pooling over the whole graph.
        vert_top = np.vstack([v.units[-1] for v in graph.get_verts()])
        edge_top = np.vstack([e.units[-1] for e in graph.get_edges()])

        return np.hstack([np.max(vert_top, axis=0), np.max(edge_top, axis=0)])

     def apply(self, graph):

        # Apply the convolutional layers.
        conv_out = self.apply_conv(graph)

        # For now, just a simple linear output at the top.
        return np.dot(conv_out, self.net_wts) + self.net_bias

    def gradient(self, graph):
        # Assumes that 'apply' had just been called so the hidden units are legit.
        
        grads = {}

        # Bias is a linear effect.
        grads['net_bias_g'] = 1.0

        # Linear output weights are also simple.
        # Should cache the end result for backprop, obviously.
        grads['net_wts_g'] = self.apply_conv(graph)

        # Grads of final max-pooling layer.
        max_vert = np.max(np.vstack([v.units[-1] for v in graph.get_verts()]), axis=0)
        max_edge = np.max(np.vstack([e.units[-1] for e in graph.get_edges()]), axis=0)

        # Handle last layer specially due to max.
        for vert in graph.get_verts():
            print vert.units[-1]

        #grads['vert_wts_g'] = [None]*(len(self.layers)-1)
        #grads['edge_wts_g'] = [None]*(len(self.layers)-1)

        for layer in range(len(self.layers)-1)[::-1]:
            pass
            # Compute edge weight gradients.
            

        return grads

    def learn(self, graph, err, rate):
        grads = self.gradient(graph)

    def checkgrad(self, graph, eps=1e-4):
        self.apply(graph)
        grads = self.gradient(graph)

        # Check the overall bias.
        self.net_bias += eps/2
        out_up = self.apply(graph)
        self.net_bias -= eps
        out_dn = self.apply(graph)
        fd_grad = (out_up - out_dn) / eps
        self.net_bias += eps/2
        print 'Overall Bias:', grads['net_bias_g'], fd_grad, np.abs(grads['net_bias_g']-fd_grad)
        
        # Check the final layer.
        print 'Overall Weights:'
        for ii in xrange(len(self.net_wts)):
            self.net_wts[ii] += eps/2
            out_up = self.apply(graph)
            self.net_wts[ii] -= eps
            out_dn = self.apply(graph)
            fd_grad = (out_up - out_dn) / eps
            self.net_wts[ii] += eps/2            
            print "\t", ii, grads['net_wts_g'][ii], fd_grad, np.abs(grads['net_wts_g'][ii]-fd_grad)

        # Loop over layers.
        #for layer in xrange(len(self.layers)-1):
            
