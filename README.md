# GANs
This is a brief go-through on some GANs. Some trained results are shown in these folders.
Three datasets are applied in these approaches: celeb_dataset, MNIST dataset and animeface dataset.
celeb_dataset and animeface dataset are for fake face generation. During the experiments, I found that animeface talks more epochs 
to generate and ususally turns to an unsatisfied result. The reason might be the colorful background fools the generator when generating fake images.
Celeb_dataset has a better result on generating faces. Number of epochs, GAN approach and dataset applied are marked at these result folders.
## GAN
## DCGAN
## WGAN
Training WGANs does not require maintaining a careful balance in training of the discriminator and the generator, and does not require a careful design of the network architecture either.
In this paper, an example is given to illustrate simple sequences of probability distributions converge under the EM distance(or Wasserstein-1) but do not converge under the other distances
and divergences which are TV, KL and JS distances and divergences. 

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
