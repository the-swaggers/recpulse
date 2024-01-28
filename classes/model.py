from classes.dtypes import Layer
from src.functions import extended_matmul
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class Model:
    """Basic model class."""

    def __init__(
        self,
        layers: [Layer],
        connection_map: [[bool]],
        precision: np.dtype = np.float32,
    ):
        self.layers = layers
        self.connections = connection_map
        self.precision = precision
        self.weights = np.zeros(shape=(self.size, self.size), dtype=object)

        for X in range(self.size):
            for y in range(self.size):
                input_size = layers[X].size
                output_size = layers[y].size
                if connection_map[X][
                    y
                ]:  # create matrix of linear transformation iff there's a connection
                    self.weights[X][y] = np.random.normal(
                        loc=0,  # mean of the distribution
                        scale=np.sqrt(
                            2 / (input_size + output_size)
                        ),  # standard deviation
                        size=(
                            output_size,
                            input_size,
                        ),  # matrix of linear transformation R^input_size -> R^output_size
                    )

        self.impulse = self.empty_impulse

    @property
    def size(self) -> int:
        """Returns numer of layers"""
        return len(self.layers)

    @property
    def empty_impulse(self):
        """Creates an empty impulse."""
        result = np.zeros(shape=self.size, dtype=object)
        for index in range(self.size):
            result[index] = np.zeros(
                shape=self.layers[index].size, dtype=self.precision
            )

        return result

    def search_by_label(self, label: str) -> int | None:
        """Returns index of the layer with the given label."""
        for index in range(self.size):
            if self.layers[index].label == label:
                return index

        return None

    def make_step(self):
        """Make a step to get new impulse."""
        new_impulse = extended_matmul(np.transpose(self.weights), self.impulse)
        self.impulse = new_impulse

    def add_impulse(self, layer_index, impulse: np.array):
        self.impulse[layer_index] += impulse

    def get_output(self, layer_index):
        return self.impulse[layer_index]

    def draw(self, save_to: str = "", name: str = "model"):
        G = nx.DiGraph()
        edges = []
        for i in range(self.size):
            for j in range(self.size):
                if self.connections[i][j]:
                    edges.append((str(i), str(j)))

        G.add_edges_from(edges)

        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos)
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_labels(G, pos)
        if save_to:
            save_to += "/"
        plt.savefig(f"{save_to}{name}.png")
