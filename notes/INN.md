### Impulse neural networks

In fact, this is a new approach to build RNNs. The Idea is to think of groups of neurons between layers as nodes of some graph and of layers as of the connections in this graph.

Word impulse comes from the second idea. To allow bidirectional connections and an ability to handle more complex graphs, we'll say that the current state of all nodes (all neurons) is impulse.

Everytime we have new input data we make a step and apply already existent impulse to calculate new one. This means that theoretically INN doesn't always require input to receive an output.

This approach gives more flexibility on model design, but mainly is just another way of building and running RNN.
