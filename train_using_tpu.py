import tensorflow as tf
import tensorflow_datasets as tds
import matplotlib.pyplot as plt
from callbacks import *
from model import *
from custom_fit import *


# Adam is going to be the optimizer for both
from tensorflow.keras.optimizers import Adam
# Binary cross entropy is going to be the loss for both 
from tensorflow.keras.losses import BinaryCrossentropy

if os.path.exists("images")==False:
  os.mkdir("images")
if os.path.exists('tmp/checkpoint')==False:
  os.mkdir('tmp/checkpoint')
  os.mkdir('tmp')

ds = tds.load("fashion_mnist",split="train")

try:
  gpus = tf.config.experimental.list_physical_devices()
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)
except:
  print("Not using GPU")


def scale_image(data):
  image = data['image']
  return image / 255


#map,cache,shuffle,batch,prefetch
ds=ds.map(scale_image)
ds=ds.cache()
ds=ds.shuffle(60000)
ds=ds.batch(128)
ds=ds.prefetch(64)

g_opt = Adam(learning_rate=0.0001) 
d_opt = Adam(learning_rate=0.00001) 
g_loss = BinaryCrossentropy()
d_loss = BinaryCrossentropy()

discriminator = build_discriminator()
generator = build_generator()

with strategy.scope():
    fashgan = FashionGAN(generator, discriminator)
fashgan.compile(g_opt, d_opt, g_loss, d_loss)

checkpoint_filepath = 'tmp/checkpoint'
#model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#    filepath=checkpoint_filepath,
#    monitor='g_loss' )


checkpoint_dir = 'tmp/checkpoint'

checkpoint_interval=4

hist = fashgan.fit(ds, 
                   epochs=2000, 
                   callbacks=[ModelMonitor(),
                   CheckpointCallback(model=fashgan, 
                                      checkpoint_dir=checkpoint_dir, 
                                      checkpoint_interval=checkpoint_interval,
                                      max_to_keep=20
                                      )
                              ]
                  )


plt.suptitle('Loss')
plt.plot(hist.history['d_loss'], label='d_loss')
plt.plot(hist.history['g_loss'], label='g_loss')
plt.legend()
plt.show()
