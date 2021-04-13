# GANs
This is a brief go-through on some GANs. Some trained results are shown in these folders.
Three datasets are applied in these approaches: celeb_dataset, MNIST dataset, maps dataset, anime dataset and animeface dataset.
celeb_dataset and animeface dataset are for fake face generation. During the experiments, I found that animeface talks more epochs 
to generate and ususally turns to an unsatisfied result. The reason might be the colorful background fools the generator when generating fake images.
Celeb_dataset has a better result on generating faces. Number of epochs, GAN approach and dataset applied are marked at these result folders.
## GAN
## DCGAN
DCGAN's generator and discriminator both apply CNN. 

***The first layer of the GAN, which takes a uniform noise distribution Z as input, could
 be called fully connected as it is just a matrix multiplication, but the result is reshaped into a 4-dimensional tensor and used as the start of the convolutional stack.***

Replace any pooling layers with stride convolutions (discriminator) and fractional-strided convolutions (generator).

Use batchnorm in both the generator and the discriminator.

Remove fully connected hidden layers for deeper architectures.

Use ReLU activation in generator for all layers except for the output, which uses Tanh.

Use LeakyReLU activation in the discriminator for all layers.

## WGAN
Training WGANs does not require maintaining a careful balance in training of the discriminator and the generator, and does not require a careful design of the network architecture either.
In this paper, an example is given to illustrate simple sequences of probability distributions converge under the EM distance(or Wasserstein-1) but do not converge under the other distances
and divergences which are TV, KL and JS distances and divergences. JS divergence has gradients issues leading to unstable training, and WGAN instead bases its loss from 
Wasserstein Distance.

In this approach, the author clamps the weights to a fixed box after each gradient update.
However, weight clipping is clearly a terrible way to enforce a Lipschitz constraints. If the clipping parameter is large,
then it can take a long time for any weights to reach their limits, thereby making it harder to train the critic till optimality.
If the clipping is small, this can easily lead to vanishing gradients when the number of 
layers is big, or batch normalization is not used. (In another word, WGAN requires that the discriminator (called the critic) must lie within the space of 1-Lipschitz functions, which the authors enforce through weight clipping.)
## WGAN_GP
WGAN sometimes can still generate poor samples or fail to converge. The reason is often due to the use of weight clipping in WGAN to enforce a Lipschitz constraint on the critic.
An alternative way to clipping weights is mentioned in WGAN_GP paper. (Penalize the norm of the critic with respect to its input.)
A soft version of the constraint with a penalty on the gradient norm for random samples.

No critic batch normalization is applied since the authors penalize the norm of the critic's gradient with respect to each input independently, and not the entire batch.

Using WGAN_GP, a strong modeling performance and stability can be reached across a variety of architectures.
## Conditional WGAN_GP
From the result of conditional WGAN_GP, we can see that every sample on fake and real are paired,
meaning that we can generate fake image from specific real image. Generative adversarial nets can be extended to a conditional model if both the generator and discriminator are 
conditioned on some extra information y.
## Pix2Pix
Here I train Pix2Pix on Maps dataset and Anime dataset. For Map dataset, I set batchsize=1 with 500 epoches. Since I could not put all the results on Github,
I make a gif to show the training process. 
![](Pix2Pix/train_map_batchsize=1_epoch_500_result/Origin_Img.gif)

