<!-- MLP with 2 hidden layers for the Discriminator and Generator -->
# Understanding the World of Generative Adversarial Networks - Part 2

In the first part of this series we understand how GANs works and what they are used for.

We already know that generative models consists of two neural networks (the Discriminator and the Generator).
So, in this article we will learn how to train and evaluate a GAN to generate handwritten digits. And, to do this, we will be using Pytorch as our framework.


## Loading Data

When we are talking about training these models, if our dataset doesn't have a considerable number of training samples, it's very hard to achieve good results for any metric. So, in order to train our model, we are going to use the [MNIST Handwritten Digits][1] dataset, which has 60.000 training samples of handwritten digits.

Lets take a look in some of these training images.

![GIF](./images/real_images.png)


```python
transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5),
                                                     std=(0.5))])
```

```python
mnist = torchvision.datasets.MNIST(root='PATH_TO_STORE_TRAINSET',
                                   train=True,
                                   transform=transform,
                                   download=True)

train_loader = torch.utils.data.DataLoader(dataset=mnist,
                                           batch_size=batch_size, 
                                           shuffle=True)
```

## Building the Model


```python
D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid())
```
```python
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh())
```

## Training Time

```python
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)
```



## Evaluation

![GIF](./images/fake_images.gif)

## Conclusions




[1]: http://yann.lecun.com/exdb/mnist/
