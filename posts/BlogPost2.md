<!-- MLP with 2 hidden layers for the Discriminator and Generator -->
# Understanding the World of Generative Adversarial Networks - Part 2

In the first part of this series we understand how GANs works and what they are used for.
We already know that generative models consists of two neural networks (the Discriminator and the Generator).
So, in this post you are going to learn how to train and use your own GAN.

When we are talking about training these models, if our dataset doesn't have a considerable number of training samples, it's very hard to achieve good results for any metric. So, in order to train our model, we are going to use the [MNIST Handwritten Digits dataset][1], which has 60.000 training samples.






[1]: http://yann.lecun.com/exdb/mnist/
