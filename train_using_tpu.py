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

def serialize_example(image, label):
    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(image).numpy()])),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()
def write_tfrecord(dataset, filename):
    with tf.io.TFRecordWriter(filename) as writer:
        for image in dataset:
            serialized_example = serialize_example(image)
            writer.write(serialized_example)

write_tfrecord(ds, 'train.tfrecord')




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

hist = fashgan.fit(ds, epochs=2000, callbacks=[ModelMonitor(),model_checkpoint_callback])


plt.suptitle('Loss')
plt.plot(hist.history['d_loss'], label='d_loss')
plt.plot(hist.history['g_loss'], label='g_loss')
plt.legend()
plt.show()
