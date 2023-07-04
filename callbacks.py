import os
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.callbacks import Callback
import tensorflow as tf


class ModelMonitor(Callback):
    def __init__(self, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.uniform((self.num_img, self.latent_dim,1))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        for i in range(self.num_img):
            img = array_to_img(generated_images[i])
            img.save(os.path.join('images', f'generated_img_{epoch}_{i}.png'))
"""
checkpoint_dir = "tmp/checkpoint"
save_interval=5
generator_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_dir + '/generator_checkpoint',
    save_weights_only=True,
    save_freq=save_interval
)
discriminator_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_dir + '/discriminator_checkpoint',
    save_weights_only=True,
    save_freq=save_interval
)
"""
class CheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self,model, checkpoint_dir, checkpoint_interval,max_to_keep,*args,**kwargs):
        super(CheckpointCallback, self).__init__(*args,**kwargs)
        self.generator_checkpoint_dir=checkpoint_dir+"/generator"
        self.discriminator_checkpoint_dir=checkpoint_dir+"/discriminator"
        self.model=model

        self.checkpoint_interval = checkpoint_interval
        self.generator_checkpoint = tf.train.Checkpoint(model=self.model.generator)
        self.generator_checkpoint_manager = tf.train.CheckpointManager(self.generator_checkpoint, self.generator_checkpoint_dir,max_to_keep)
        self.discriminator_checkpoint = tf.train.Checkpoint(model=self.model.discriminator)
        self.discriminator_checkpoint_manager = tf.train.CheckpointManager(self.discriminator_checkpoint, self.discriminator_checkpoint_dir,max_to_keep)


    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.checkpoint_interval == 0:
            self.discriminator_checkpoint_manager.save(checkpoint_number=epoch)
            print(f"Checkpoint saved at epoch {epoch + 1}")
            self.generator_checkpoint_manager.save(checkpoint_number=epoch)
            print(f"Checkpoint saved at epoch {epoch + 1}")
