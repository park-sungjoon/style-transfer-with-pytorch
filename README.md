# style-transfer-with-pytorch
Implementations of some style transfer techniques.

## pix2pix
Simple code for [pix2pix](https://arxiv.org/abs/1611.07004) (see this [page](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) for the pytorch implementation provided by the authors).
Download data from this [page](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/) and unzip it to a folder containing both the folders 'datasets' and the folder 'pix2pix'.
To train and sample images during the training, run `python train.py --epochs=200`.
Note that we sample every 10 epochs and at the end of the training.
To save the model, pass on the option `--save-model`.

Below, we show results obtained after training on the facades dataset.
From top to bottom, we show the annotaed image, target image, generated image.
<p align="center">
  <img src="pix2pix/visualize/val/x200.png" width="800"\> <br>
  <img alt="" src="pix2pix/visualize/val/y200.png" width="800"\> <br>
  <img alt="" src="pix2pix/visualize/val/y_G200.png" width="800"\>
</p>

## cycleGAN
Implementation of cycleGAN (see this [page](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) for the pytorch implementation provided by the authors).

For facades, download daa from this [page](http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/) (see cycleGAN_dataset for note on how the data should be organized).
For the loss function, we use the GAN loss and the cycle loss as defined in the cycleGAN paper.
We set the number of resnet layers to be 9.
Below, we show some generated data:
<p align="center">
  <img src="cycleGAN/visualize/facades/val/x_G200.png" width="800"\> <br>
  <img src="cycleGAN/visualize/facades/val/y_G200.png" width="800"\> <br>
</p>

We also tried translation between photos and animation.
For photos, we combined the flickr dataset provided by the authors of cycleGAN with some of the FFHQ dataset.
For animation dataset, we sampled from some animated movies from Studio Ghibli (Spirited Away, Castle in the Sky, Howl's Moving Castle).
<p align="center">
  <img src="cycleGAN/visualize/ani2photo/val/x_G200.png" width="800"\> <br>
  <img src="cycleGAN/visualize/ani2photo/val/y_G200.png" width="800"\> <br>
</p>

For training, we included the identity loss described in the cycleGAN paper 
(run `cycleGAN_train.py` with the optional argument `--pix-loss`). 
Without this loss, we found that the generator can invert light and dark, most likely because pixel distribution is easy to learn.
We implement this loss using the class down_sampler in cycleGAN_model.py.
To allow more degree of freedom for the generator, one can turn on the blur&pool option.
Below, we show the result of using blur&pool option (we weighed the pix_loss by a factor of 10 instead of 5 for this result)
<p align="center">
  <img src="cycleGAN/visualize/ani2photo/val/x_G200-downsampled-pixloss.png" width="800"\> <br>
</p>

As can be seen, this worsens image quality. 
The most prominent feature is that the barcode-like artifacts become worse (note that such artifacts are present with the original setting of cycleGAN as well).
Such oscillatory behavior is reminicent of the [oscillatory behavior](https://arxiv.org/abs/1904.11486) of CNN classifiers that does not use anti-ailiasing in downsampling and upsampling layers.
The identity loss (i.e. no downsampling) tries to preserve pixel at each position in space, so that oscillatory behavior is penalized.
However, downsampling allows more freedom, and it could allowing the oscillatory behavior to become materialize more prominently.

## Things to do
* Test whether anti-ailiasing will alleviate artifacts described above.
* Try contrastive unpaired translation ([CUT](https://arxiv.org/abs/2007.15651))
