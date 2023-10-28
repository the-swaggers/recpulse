# What is this library about?

Probably everyone who reads this is already familiar with how CNNs (Convolutional Neural Networks) work. If not, it's highly recommended to learn them first.

In this library we want to generalize this approach by looking at layers as nodes of some graph. Connections of this graph will represent what layers are connected and in what way.

It allows us to build way more complicated structures, but as we are on the early stage of development, good results can't be guaranteed.

As graphs can be cyclic, the first input might affect next outputs, so we will introduce new term: **impulse**.

### What is impulse?

As we look at our model as at graph, we'll can an impulse a set of values in each neuron at the particular moment. 

### Why this approach?

It allows us to store all the results of the previous runs elegantly.

### How to train this model?

Well, as we are now only working on this, we can't know how to train this model. Probably, The algorythm will be similar to what CNNs use, but as we don't have any proofs of its efficiency in this particular case, we can't state that it'll be used. So, we have a yet to solve it.

### Reason why the library's naming

As we get a new **impulse** from an old one, we can call it recurrent.

Note: We intend to call this type of NNs - INN (Impulse Neural Network).
