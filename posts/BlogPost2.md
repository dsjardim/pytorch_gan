<!-- MLP with 2 hidden layers for the Discriminator and Generator -->
# Understanding the World of Generative Adversarial Networks - Part 2

In the first part of this series we understand how GANs works and what they are used for.

We already know that generative models consists of two neural networks (the Discriminator and the Generator).
So, in this article we will learn how to train and evaluate a GAN to generate handwritten digits.

We will be using Pytorch as the framework and the full code is available in [this GitHub repository](https://github.com/dsjardim/pytorch_gan).


## Loading Data

When we are talking about training these models, if our dataset doesn't have a considerable number of training samples, it's very hard to achieve good results for any metric. So, in order to train our model, we are going to use the [MNIST Handwritten Digits][1] dataset, which has 60.000 training samples of handwritten digits.

Lets take a look in some of these training images.

![RealImages](./images/real_images.png)

Pytorch provides an easy way to download the training samples using a few lines of code.
But before downloading the data, we have to define some transformations we need to apply on our data before feeding it into the training pipeline. We do this using the ```torchvision.transforms```. It basically transform each image to tensor format and normalize it.

```python
transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5),
                                                     std=(0.5))])
```

Now, with the following code snippet, we download the data, apply the transformations in it and load it to DataLoader, which split the data into batches and provides iterators over the dataset.

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

### Training the Discriminator

```python
outputs = D(images)
d_loss_real = criterion(outputs, real_labels)
real_score = outputs

z = torch.randn(batch_size, latent_size).to(device)
z = Variable(z)
fake_images = G(z)

outputs = D(fake_images)
d_loss_fake = criterion(outputs, fake_labels)
fake_score = outputs

d_loss = d_loss_real + d_loss_fake
```

### Training the Generator

```python
z = torch.randn(batch_size, latent_size).to(device)
z = Variable(z)
fake_images = G(z)
outputs = D(fake_images)

g_loss = criterion(outputs, real_labels)
```

## Evaluation

```python
D.load_state_dict(torch.load('./save/D--300.pth'))
G.load_state_dict(torch.load('./save/G--300.pth'))

latent_size = 64
batch_size = 25

z = torch.randn(batch_size, latent_size).to(device)
fake_images = G(z)

fake_images_np = fake_images.cpu().detach().numpy()
fake_images_np = fake_images_np.reshape(fake_images_np.shape[0], 28, 28)
```

![GIF](./images/fake_images.gif)

## Conclusions




[1]: http://yann.lecun.com/exdb/mnist/
