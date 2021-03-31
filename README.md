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
## WGAN_GP
## Conditional WGAN_GP
From the result of conditional WGAN_GP, we can see that every sample on fake and real are paired,
meaning that we can generate fake image from specific real image.