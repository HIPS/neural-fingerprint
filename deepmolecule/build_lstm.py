import numpy as np
from funkyyak import grad
from util import VectorParser

def squash(x):
    return 0.5*(np.tanh(x) + 1)   # Output ranges from 0 to 1.

def activations(input, state, weights):
    cat_state = np.concatenate((input, state, np.ones((input.shape[0],1))), axis=1)
    return squash(np.dot(cat_state, weights))

def build_lstm(seq_width, state_size, output_size, l2_penalty=0.0):
    parser = VectorParser()
    parser.add_shape('change', (seq_width + state_size + 1, state_size))
    parser.add_shape('gate',   (seq_width + state_size + 1, state_size))
    parser.add_shape('keep',   (seq_width + state_size + 1, state_size))
    parser.add_shape('output', (state_size, output_size))

    def update_lstm(input, state, change_weights, gate_weights, keep_weights):
        """One iteration of an LSTM layer without an output."""
        change = activations(input, state, change_weights)
        gate   = activations(input, state, gate_weights)
        keep   = activations(input, state, keep_weights)
        return state * keep + gate * change

    def compute_hiddens(weights_vect, seqs):
        """Goes from right to left, updating the state."""
        weights = parser.new_vect(weights_vect)
        num_seqs = seqs.shape[1]
        state = np.zeros((num_seqs, state_size))
        for cur_input in seqs:  # Iterate over time steps.
            state = update_lstm(cur_input, state,
                                weights['change'], weights['gate'], weights['keep'])
        return state

    def predictions(weights_vect, seqs):
        weights = parser.new_vect(weights_vect)
        return np.dot(compute_hiddens(weights_vect, seqs), weights['output'])

    def loss(weights, seqs, targets):
        log_lik = -np.sum((predictions(weights, seqs) - targets)**2)
        log_prior = -l2_penalty * np.dot(weights, weights)
        return (-log_prior - log_lik) / targets.shape[0]

    return loss, grad(loss), predictions, compute_hiddens, parser


def build_rnn(seq_width, state_size, output_size, l2_penalty=0.0):
    parser = VectorParser()
    parser.add_shape('update', (seq_width + state_size + 1, state_size))
    parser.add_shape('output', (state_size, output_size))

    def compute_hiddens(weights_vect, seqs):
        """Goes from right to left, updating the state."""
        weights = parser.new_vect(weights_vect)
        num_seqs = seqs.shape[1]
        state = np.zeros((num_seqs, state_size))
        for cur_input in seqs:  # Iterate over time steps.
            state = activations(cur_input, state, weights['update'])
        return state

    def predictions(weights_vect, seqs):
        weights = parser.new_vect((weights_vect))
        return np.dot(compute_hiddens(weights.vect, seqs), weights['output'])

    def loss(weights_vect, seqs, targets):
        log_lik = -np.sum((predictions(weights_vect, seqs) - targets)**2)
        log_prior = -l2_penalty * np.dot(weights_vect, weights_vect)
        return (-log_prior - log_lik) / targets.shape[0]

    return loss, grad(loss), predictions, compute_hiddens, parser
