#Importing the sequential API for building our Discriminator and Generator
from tensorflow.keras.models import Sequential,Model
#Importing the layers for building our Discriminator and Generator
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Reshape, LeakyReLU, Dropout, UpSampling2D


def build_discriminator():
  model=Sequential()

  model.add(Conv2D(32,5,input_shape=(28,28,1)))
  model.add(LeakyReLU(0.2))
  model.add(Dropout(0.4))

  model.add(Conv2D(64,5))
  model.add(LeakyReLU(0.2))
  model.add(Dropout(0.4))

  model.add(Conv2D(128,5))
  model.add(LeakyReLU(0.2))
  model.add(Dropout(0.4))

  model.add(Conv2D(256,5))
  model.add(LeakyReLU(0.2))
  model.add(Dropout(0.4))


  model.add(Flatten())
  model.add(Dropout(0.4))
  model.add(Dense(1,activation="sigmoid"))

  return model

def build_generator():
  model=Sequential()

  model.add(Dense(7*7*128,input_dim=128))
  model.add(LeakyReLU(0.2))
  model.add(Reshape((7,7,128)))

  #Upsampling layer 1
  model.add(UpSampling2D())
  model.add(Conv2D(128, 5,padding='same'))
  model.add(LeakyReLU(0.2))

  #Upsampling layer 2
  model.add(UpSampling2D())
  model.add(Conv2D(128, 5,padding='same'))
  model.add(LeakyReLU(0.2))

  model.add(Conv2D(128, 4,padding='same'))
  model.add(LeakyReLU(0.2))

  model.add(Conv2D(128, 4,padding='same'))
  model.add(LeakyReLU(0.2))

  model.add(Conv2D(1, 4,padding='same',activation="sigmoid"))


  return model

class Discriminator(Model):
  def __init__(self,  *args, **kwargs):
    # Pass through args and kwargs to base class 
    super().__init__(*args, **kwargs)
    self.conv1=Conv2D(32,5,input_shape=(28,28,1))
    self.activation=LeakyReLU(0.2)
    self.conv2=Conv2D(64,5)
    self.conv3=Conv2D(128,5)
    self.conv4=Conv2D(256,5)
    self.dense=Dense(1,activation="sigmoid")
    self.dropout=Dropout(0.4)
    self.flatten = Flatten()
    

  def call(self,inputs,training=False):
    x=self.conv1(inputs)
    x=self.activation(x)
    x=self.dropout(x)
    x=self.conv2(x)
    x=self.activation(x)
    x=self.dropout(x)
    x=self.conv3(x)
    x=self.activation(x)
    x=self.dropout(x)
    x=self.conv4(x)
    x=self.activation(x)
    x=self.dropout(x)
    x=self.flatten(x)
    x=self.dropout(x)
    x=self.dense(x)
    
    
    return x

class Generator(Model):
  def __init__(self,  *args, **kwargs): 
    super().__init__(*args, **kwargs)
    self.dense=Dense(7*7*128,input_dim=128)
    self.activation=LeakyReLU(0.2)
    self.reshape=Reshape((7,7,128))
    self.upsample=UpSampling2D()
    self.conv1=Conv2D(128, 5,padding='same')
    self.conv2=Conv2D(128, 4,padding='same')
    self.conv3=Conv2D(1, 4,padding='same',activation="sigmoid")

  def call(self,inputs,training=False):
    x=self.dense(inputs)
    x=self.activation(x)
    x=self.reshape(x)
    x=self.upsample(x)
    x=self.conv1(x)
    x=self.activation(x)
    x=self.upsample(x)
    x=self.conv1(x)
    x=self.activation(x)
    x=self.upsample(x)
    x=self.conv2(x)
    x=self.activation(x)
    x=self.upsample(x)
    x=self.conv2(x)
    x=self.activation(x)
    x=self.conv3(x)
    return x
    
