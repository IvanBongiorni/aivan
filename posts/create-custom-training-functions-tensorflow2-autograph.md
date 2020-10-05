# Create Custom Training Functions in TensorFlow 2 (with Autograph too!)

Summary:

- Training models from TensorFlow 1.x to 2.x
- Create a custom training function
- Make custom training fast enough with Autograph

This post is about how to implement **custom training** of Neural Networks in **TensorFlow 2**, and how to make it run faster with a new feature called **Autograph**.
If you are interested on the topic, I suggest you to take a more "in depth" look at my [Tutorial Notebooks on TensorFlow 2](https://github.com/IvanBongiorni/TensorFlow2.0_Notebooks).
They are all based on custom training functions, and they can be seen as a more practical application of the topics presented here.

<br/>


## Training models from TensorFlow 1.x to 2.x

TensorFlow 1.x was based on symbolic programming, worked very differently from how we usually think about code.
In fact, you couldn't simply write down an operation (let's say, an addition like c=a+b) and observe its execution on the spot,
instead, you had to first create a **computational graph**, i.e. an empty structure of abstract computations, and in a second phase push tensors through it... making a *tensor flow*.

Symbolic programming was quite cool (IMHO) but very uncomfortable to write and complicated to debug.
For that reason (and the competition that started with pyTorch) the newer TensorFlow 2 was characterized by **eager execution** instead, i.e. it works just like in plain Python.

Additionally, **Keras** is no more just an additional layer, built on top of the main library, but it became an essential part of it.
Every architecture in fact now must be built with Keras' layers and syntax.

In fact, canonical model training in TensorFlow 2 is now done with:

```
model.compile()
history = model.fit(X_train, Y_train)
```

which is the good ol' Keras syntax.

That's good an all, but what if you need more control?
What if you need a more articulated training mechanism for a unique problem that requires a unique solution?
What if you'd like to 
Or what if you are just a curious student that wants to have a better understanding of how Neural Networks are trained?

In all these cases, a simple `model.fit()` is not enough, you need to write your own **custom training function**.
Keras is super easy and confortable to use, but you're not going to lear Deep Learning with it.

<br/>


## Create a custom training function

In a custom training function you're going to need **for loops**.
For each iteartion, you need to follow these steps:

First, define a Loss and an optimizer.
That's easy, no need to explain that.

Second, record the gradient that you take from computing the loss value.
This can be done with `tf.GradientTape()`, this is its purpose.
A `tf.GradientTape()` object records the gradients (i.e. computes the impact of each weight to the final loss value).

Third, use that gradient to update the weights (through the optimizer).

Let's say you want to train a model to solve a *classification* problem, after defining Loss and optimizer:

```
loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.optimizers.Adam(learning_rate = 0.0001)
```

a custom training step would look like that:

```
with tf.GradientTape() as tape:
    current_loss = loss(model(X_batch), y_batch)
        
gradients = tape.gradient(current_loss, model.trainable_variables)
optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

A `tf.GradientTape()` object is created as tape, to "take a look" at the value of loss generated. 
Later, the gradient is computed using `tape.gradient` on the `model.trainable_variables`, 
and final weight update is applied with `optimizer.apply_gradients`.

I happened to use. Here are two examples I found interesting:

- Training a CNN with custom data augmentation: 
I was working on an image classification task, I wanted to implement my own, personalized data augmentation pipeline (I used a very nice library called albumentations, check it out).
It's not possible to call a personalized data augmentatino function with a basic model.fit() type of command.
Therefore, I packed the code above in a custom train_step() function, I wrote my own training loop that called it iteratively together with my own personalized data augmentation function.

- I trained a GAN for a personal project: a [Conv-Recurrent GAN for the imputation of missing data in time series](https://github.com/IvanBongiorni/GAN-RNN_Timeseries-imputation).
The complex interaction between Generator and Discriminator made custom training extremely useful, it gave me complete freedom and flexibility in the implementation.

I had a lot of fun working on these projects.

<br/>


## Make custom training fast enough with Autograph

Yeah, I know what you might be thinking now: Python loops are *terribly slow*.
And you'd be right!
Python loops are an order of magnitude slower than C++ loops.

Fortunately, there is a little tool called **Autograph** that allows you to significantly boost your code: **Autograph**, and more specifically the **@tf.function** decorator.

What is it?

As you know, TensorFlow counts a number of built-in functions (they are usually called *ops*).

Every TensorFLow computational graph (which is still present, just as in the 1.x version, you just can't see it anymore) is a composition of many of these ops.
Since the TensorFlow kernel is written in C++, once your Python code gets compiled as a computational graph and executed very quickly.

The goal of the `@tf.function` decorator is to transform your own Python functions into TensorFlow ops.

You just take the `@tf.function` decorator and put it on top of your training function.
Like that:

```
@tf.function
def train_step():
    with tf.GradientTape() as tape:
        current_loss = loss(model(X_batch), y_batch)
        
    gradients = tape.gradient(current_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables)
    return current_loss
```

Autograph allows you to turn any function of your choice into a TensorFlow graph, making its execution orders of magnitude faster.
Even functions that have nothing to do with model training can be optimized with `@tf.function`.

However, some conditions must be met for Autograph to work.

The input of you function must be a numpy `array`.

Every function with `@tf.function` on top must be composed of other native Python functions, or TensorFlow ops (you can't use numpy functions, for example).

This is less problematic than what it might look like.
Many TensorFlow functions are analogous to numpy ones.
You can use, for example, `tf.where()` instead of `np.where()`, or write `tf.cast(x, tf.float32)` instead of `x.astype(np.float32)`.

And remember: if you have a loop, don't use `range()` but `tf.range()`; Autograph will automatically translate it into a `tf.while_loop()`.

And that's it!
Hopefully, this post will help you write better, more advanced training functions and optimize your TensorFlow code as much as possible.
As I already said above, I wrote a number of [Tutorial Notebooks on TensorFlow 2](https://github.com/IvanBongiorni/TensorFlow2.0_Notebooks) with the use of custom training and Autograph.

Enjoy! (Hopefully)


