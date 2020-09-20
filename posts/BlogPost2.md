<!-- MLP with 2 hidden layers for the Discriminator and Generator -->
# Understanding the World of Generative Adversarial Networks - Part 2

In the first part of this series we understand how GANs works and what they are used for.

We already know that generative models consists of two neural networks (the Discriminator and the Generator).
So, in this article we will learn how to train and evaluate a GAN to generate handwritten digits.

We will be using Pytorch as the framework and the full code is available in [this GitHub repository](https://github.com/dsjardim/pytorch_gan).


## Loading the Data

When we are talking about training these models, if our dataset doesn't have a considerable number of training samples, it's very hard to achieve good results for any metric. So, in order to train our model, we are going to use the [MNIST Handwritten Digits][1] dataset, which has 60.000 training samples of handwritten digits. And each image have its resolution defined as 28 x 28 (width, height). 

Lets take a look in some of these training images.

<img src="./images/real_images.png" alt="RealImages" width="400"/>


Pytorch provides an easy way to download the training samples using a few lines of code.
But before downloading the data, we have to define some transformations we need to apply on our data before feeding it into the training pipeline. We do this using the ```torchvision.transforms```. It basically transform each image to tensor format and normalize it.

```python
transform = transforms.Compose([transforms.ToTensor(),
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

Both discriminator and generator are Multilayer Perceptrons (MLPs) with two hidden layers.

In Pytorch, the ```torch.nn``` module allow us to build both neural networks very easelly.
Also, the ```nn.Sequential``` wraps the layers in the network.

Lets look at the following code snippet, which builds the Discriminator of our GAN.

```python
D = nn.Sequential(
    nn.Linear(784, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 1),
    nn.Sigmoid())
```

As you can see, there are three **Linear** layers with **leaky ReLU** activation function. 
The first two are called **hidden layers** and the last one is our **output layer**.
Also, the output layer is followed by a ```Sigmoid``` function because we want to calculate the probability of an image being real or fake.

Probably, you must be wondering why do we have 784 units in the first layer. Good! 
It is because we flatten out each image before sending it inside the neural network. (28 x 28 = 784)

The generator have the same architecture. But different parameters for each layer. 

Lets take a look at the code used to build it.

```python
G = nn.Sequential(
    nn.Linear(64, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 784),
    nn.Tanh())
```

You can see that the parameters appear to be almost reversed compared to the discriminator.

It make sense, because here we are not classifying a given image, we are trying to create a new one.
So, in the first layer, we receive a random noise vector as input. 
And the following layers try to fit this input to generate a image that look like our training data. 


## Training Time

```python
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)
```

```python
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(train_loader):
        images = images.view(batch_size, -1).to(device)
        
        real_labels = torch.ones(batch_size, 1).to(device) ## 1 for real samples
        fake_labels = torch.zeros(batch_size, 1).to(device) ## 0 for fake samples

        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #
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
        
        reset_grad()
        d_loss.backward()
        d_optimizer.step()

        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #
        z = torch.randn(batch_size, latent_size).to(device)
        z = Variable(z)
        fake_images = G(z)
        outputs = D(fake_images)

        g_loss = criterion(outputs, real_labels)

        reset_grad()
        g_loss.backward()
        g_optimizer.step()
```


## Evaluation and Conclusions

We trained our GAN through 300 epochs and after that we evaluate our model.

During training, we stored information about the Loss for both discriminator and generator. Also, we got information about the Accuracy of the discriminator for classifying images as real or fake.

Lets take a look at this data in form of graphs.

<img src="./images/acc_loss.png" alt="AccLoss" width="800"/>

First, analyzing the accuracy of the discriminator. It starts classifying real images with high accuracy, but after a while, it starts to make mistakes more often, which is expected. At the same time, for fake images classification, the discriminator start the training with high accuracy as well (close to 0 because it's a fake image) and after a while, it starts to make mistakes because the generator are creating images very close to the training images. 

Now, lets talk about the loss of both discriminator and generator.
We can see that the loss of the generator decreases and the loss of the discriminator increases.
This is what we want! Because it indicates that our generator learned to create images that reproduce the training data. And this is why the discriminator starts to make mistakes. It no longer knows whether a given image is real or fake.

The following GIF present the learning process of the generator.

<img src="./images/fake_images.gif" alt="GIF" width="400"/>

For an evaluation, we test our GAN in order to generate some handwritten digits. And the result are the following

<img src="./images/generated_images.png" alt="GenImages" width="300"/>

As you can see, there is no difference between these generated images from those from the MNIST dataset we saw in the beginning of this article.

Therefore, it proves that we can use GANs to augment our dataset. Here we have worked with a dataset of images, but we can train and use this models for any kind of dataset. The only constraint is that the dataset have to be a large number of training samples. 


[1]: http://yann.lecun.com/exdb/mnist/
