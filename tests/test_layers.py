import classes.layers as layers


class TestLayerClass:
    def test_init(self):
        layers.Dense((4,))
        layers.Dense(4)
