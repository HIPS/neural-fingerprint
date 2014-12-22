
from funkyyak import grad, numpy_wrapper as np

from util import WeightsParser

def nonlinearity(units, weights):
    """Should have output ranging from 0 to 1."""
    return 0.5*(np.tanh(np.dot(units, weights)) + 1)

def squash(x):
    return 0.5*(np.tanh(x) + 1)

def lstm(gate_weights, change_weights, keep_weights, input, state):
    """Implements one iteration of an LSTM node without an output."""
    bias = np.ones((1))#state.shape[0], 1))
    combined = np.concatenate((input, state, bias), axis=0)
    change = nonlinearity(combined, change_weights)
    gate   = nonlinearity(combined, gate_weights)
    keep   = nonlinearity(combined, keep_weights)
    return state * keep + gate * change


one_hot = lambda x, K : np.array(x[:,None] == np.arange(K)[None, :], dtype=int)

def build_lstm_rnn(input_size, state_size, l2_penalty=0.0):

    parser = WeightsParser()
    parser.add_weights('gate',   (input_size + state_size + 1, state_size))
    parser.add_weights('change', (input_size + state_size + 1, state_size))
    parser.add_weights('keep',   (input_size + state_size + 1, state_size))

    def apply_lstm_to_seq(weights, seq):
        """Goes from right to left, updating the state."""
        state = np.zeros(state_size)
        gate_weights = parser.get(weights,   'gate')
        change_weights = parser.get(weights, 'change')
        keep_weights = parser.get(weights, 'keep')
        input_matrix = one_hot(seq, input_size)
        for cur_input in input_matrix:
            state = lstm(gate_weights, change_weights, keep_weights, cur_input, state)
        return state

    def apply_lstm_to_seqs(weights, seqs):
        states = np.zeros((seqs.shape[0], state_size))
        for ix, seq in enumerate(seqs):
            states[ix, :] = apply_lstm_to_seq(weights, seq)
        return states

    parser.add_weights('output', (state_size, input_size))

    def predictions(weights, seqs):
        """Go from the fixed-size representation to a prediction."""
        return np.dot(apply_lstm_to_seqs(weights, seqs), parser.get(weights, 'output'))

    def loss(weights, seqs, targets):
        log_lik = np.sum((predictions(weights, seqs) - one_hot(targets, input_size))**2)
        log_prior = -l2_penalty * np.dot(weights, weights)
        return -log_prior - log_lik

    return loss, grad(loss), predictions, apply_lstm_to_seqs, parser
