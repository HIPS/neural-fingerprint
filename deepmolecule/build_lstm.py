from funkyyak import grad, numpy_wrapper as np
from util import WeightsParser

def squash(x):
    return 0.5*(np.tanh(x) + 1)   # Output ranges from 0 to 1.

def nonlinearity(input, state, weights):
    # Divide up weight matrix.
    num_inputs = input.shape[1]
    num_state = state.shape[1]
    input_weights = weights[0:num_inputs, :]
    state_weights = weights[num_inputs:num_inputs+num_state, :]
    bias = weights[-1,:]
    inner_sum = np.dot(input, input_weights) \
              + np.dot(state, state_weights) + bias
    return squash(inner_sum)

def update_lstm(weights, parser, input, state):
    """One iteration of an LSTM node without an output."""
    gate_weights   = parser.get(weights, 'gate')
    change_weights = parser.get(weights, 'change')
    keep_weights   = parser.get(weights, 'keep')
    change = nonlinearity(input, state, change_weights)
    gate   = nonlinearity(input, state, gate_weights)
    keep   = nonlinearity(input, state, keep_weights)
    return state * keep + gate * change

def one_hot(x, K):
    return np.array(x[:,None] == np.arange(K)[None, :], dtype=int)

def build_lstm_rnn(input_size, state_size, output_size=1, l2_penalty=0.0):
    parser = WeightsParser()
    parser.add_weights('gate',   (input_size + state_size + 1, state_size))
    parser.add_weights('change', (input_size + state_size + 1, state_size))
    parser.add_weights('keep',   (input_size + state_size + 1, state_size))
    parser.add_weights('output', (state_size, output_size))

    def compute_hiddens(weights, seqs):
        """Goes from right to left, updating the state."""
        num_seqs = seqs.shape[0]
        state = np.zeros((num_seqs, state_size))
        for cur_input in seqs.T:  # Iterate over time steps.
            inputs_onehot = one_hot(cur_input, input_size)
            state = update_lstm(weights, parser, inputs_onehot, state)
        return state

    def predictions(weights, seqs):
        return np.dot(compute_hiddens(weights, seqs), parser.get(weights, 'output'))

    def loss(weights, seqs, targets):
        log_lik = -np.sum((predictions(weights, seqs) - targets)**2)
        log_prior = -l2_penalty * np.dot(weights, weights)
        return -log_prior - log_lik

    return loss, grad(loss), predictions, compute_hiddens, parser


def update_rnn(weights, parser, input, state):
    """One iteration of a standard RNN node without an output."""
    change_weights = parser.get(weights, 'update')
    return nonlinearity(input, state, change_weights)

def build_vanilla_rnn(input_size, state_size, output_size=1, l2_penalty=0.0):
    parser = WeightsParser()
    parser.add_weights('update', (input_size + state_size + 1, state_size))
    parser.add_weights('output', (state_size, output_size))

    def compute_hiddens(weights, seqs):
        """Goes from right to left, updating the state."""
        num_seqs = seqs.shape[0]
        state = np.zeros((num_seqs, state_size))
        for cur_input in seqs.T:  # Iterate over time steps.
            inputs_onehot = one_hot(cur_input, input_size)
            state = update_rnn(weights, parser, inputs_onehot, state)
        return state

    def predictions(weights, seqs):
        return np.dot(compute_hiddens(weights, seqs), parser.get(weights, 'output'))

    def loss(weights, seqs, targets):
        log_lik = -np.sum((predictions(weights, seqs) - targets)**2)
        log_prior = -l2_penalty * np.dot(weights, weights)
        return -log_prior - log_lik

    return loss, grad(loss), predictions, compute_hiddens, parser
