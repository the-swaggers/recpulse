# What is this library about?

This library will operate the following term: **impulse**. It will represent all impulses in the model. In fact, impulse I(x<sub>i</sub>, θ) might be recursive. It makes INN (Impulse Neural Network) be similar to RNN. However, there are key differences:

* INN allows to operate multiple inputs/outputs more flexible.
* INN has more variants to crate the recursion.
* INN will support external functions that will make will bring more opportunities to structure a model.

The whole motto of INN: `Generalize and chill`

At this moment, library is just making a generalization on neural networks. Roughly speaking, we can say

    CNN ⊆ RNN ⊆ INN

At least it's how I see it.

## The beauty of math

As there is a beautiful algorythm for step from one layer to another: matrix multiplication, there is a willing to generalize this.

There is almost nothing "good" that can't be stated for more complex tasks. In this case, we can represent layers as vertexes and transformations between them as edges. It allows us to create a matrix that will have the following structure:

* If there are n layers the matrix will be of size n×n.
* All layers are iterated and named l<sub>i</sub> 
* The element of matrix in i<sup>th</sup> column and j<sup>th</sup> will be either
  * None is there is no connection from l<sub>i</sub> to l<sub>j</sub>
  * Matrix of transformation from l<sub>i</sub> to l<sub>j</sub> otherwise

This will allow us to make a step to the next impulse by adding input to the previous one and then multiplying it by this matrix.

## Current purpose

At the moment the main goal of the library is to build a convenient tool to build various complex NNs.

Inspiration went from thoughts that if we want to develop something really thinking, we have to make ot work without any inputs. Right now, we consider it as a possible future application of this library, but there are many challenges to overcome before ths can be implemented.