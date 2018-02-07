import tensorflow as tf
from tensorflow.python.util import nest

class DropoutWrapper(tf.contrib.rnn.RNNCell):
    """Operator adding dropout to inputs and outputs of the given cell."""

    def __init__(self, cell, input_keep_prob=1.0, output_keep_prob=1.0,
                 seed=None):

        if not isinstance(cell, tf.contrib.rnn.RNNCell):
            raise TypeError("The parameter cell is not a RNNCell.")
        if (isinstance(input_keep_prob, float) and
                not (input_keep_prob >= 0.0 and input_keep_prob <= 1.0)):
            raise ValueError("Parameter input_keep_prob must be between 0 and 1: %d"
                             % input_keep_prob)
        if (isinstance(output_keep_prob, float) and
                not (output_keep_prob >= 0.0 and output_keep_prob <= 1.0)):
            raise ValueError("Parameter output_keep_prob must be between 0 and 1: %d"
                             % output_keep_prob)
        self._cell = cell
        self._input_keep_prob = input_keep_prob
        self._output_keep_prob = output_keep_prob
        self._seed = seed

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        """Run the cell with the declared dropouts."""
        if (not isinstance(self._input_keep_prob, float) or
                    self._input_keep_prob < 1):
            inputs = tf.nn.dropout(inputs, self._input_keep_prob, seed=self._seed)
        output, new_state, input_gate, forget_gate, output_gate = self._cell(inputs, state, scope)
        if (not isinstance(self._output_keep_prob, float) or
                    self._output_keep_prob < 1):
            output = tf.nn.dropout(output, self._output_keep_prob, seed=self._seed)
        return output, new_state, input_gate, forget_gate, output_gate

def rnn(cell, inputs, initial_state, scope=None):
    """Creates a recurrent neural network specified by RNNCell `cell`."""

    if not isinstance(cell, tf.contrib.rnn.RNNCell):
        raise TypeError("cell must be an instance of RNNCell")
    if not nest.is_sequence(inputs):
        raise TypeError("inputs must be a sequence")
    if not inputs:
        raise ValueError("inputs must not be empty")

    outputs = []
    input_gates = []
    forget_gates = []
    output_gates = []
    # Create a new scope in which the caching device is either
    # determined by the parent scope, or is set to place the cached
    # Variable using the same placement as for the rest of the RNN.
    with tf.variable_scope(scope or "RNN") as varscope:
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)

        state = initial_state

        for time, input_ in enumerate(inputs):
            if time > 0: varscope.reuse_variables()
            call_cell = lambda: cell(input_, state)
            output, state, input_gate, forget_gate, output_gate = call_cell()
            outputs.append(output)
            input_gates.append(input_gate)
            forget_gates.append(forget_gate)
            output_gates.append(output_gate)

    return (outputs, state, input_gates, forget_gates, output_gates)
