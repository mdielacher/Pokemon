import numpy as np
import tensorflow as tf
from keras import layers, models
import pandas as pd
np.random.seed(42)

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


class GAN():

    def __init__(self, data, noise, epochs) -> None:
       self.data = data
       self.noise = noise
       self.epochs = epochs

    def create_generator(self):
        model = models.Sequential(name="generator_model")
        model.add(layers.Dense(15, activation='relu',input_dim=self.data.shape[1]))
        model.add(layers.Dense(30, activation='relu'))
        model.add(layers.Dense(self.data.shape[1], activation='tanh'))
        return model
    
    
    def create_discriminator(self):
        model = models.Sequential(name="discriminator_model")
        model.add(layers.Dense(25, activation='relu',input_dim=self.data.shape[1]))
        model.add(layers.Dense(50, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
        return model
    
    def compile(self, generator, discriminator):
        discriminator.trainable = False
        generator.trainable = True
        model = models.Sequential(name="GAN")
        model.add(generator)
        model.add(discriminator)
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model

    def train(self, generator, discriminator, gan):
        for epoch in range(self.epochs):
            
            # Train the discriminator
            generated_data = generator.predict(self.noise)
            generated_data = generated_data.astype(np.float32)

            # Prepare real and synthetic labels
            real_labels = np.ones((self.data.shape[0], 1), dtype=np.float32)
            fake_labels = np.zeros((self.data.shape[0], 1), dtype=np.float32)
            labels = np.concatenate([real_labels, fake_labels])
            X = np.concatenate([self.data, generated_data])
            discriminator.trainable = True
            d_loss , _ = discriminator.train_on_batch(X, labels)

            # Train the generator
            g_loss = gan.train_on_batch(self.noise, np.ones(self.data.shape[0]))


            print('>%d, d1=%.3f, d2=%.3f' %(epoch+1, d_loss, g_loss))

        return generator
    

# def main():
    
#     df_pokemon= pd.read_csv("./data/Pokemon.csv")
#     df_pokemon = df_pokemon.drop(columns=['Name', 'Total'], axis=1)
#     df_pokemon.head()
#     noise = np.random.normal(0, 1, df_pokemon.shape) 
#     gan = GAN(data=df_pokemon, noise=noise, epochs=2)
#     generator = gan.create_generator()
#     discriminator = gan.create_discriminator()
#     gan_model = gan.compile(generator=generator, discriminator=discriminator)
#     trained_gan = gan.train(generator=generator,discriminator=discriminator, gan=gan)

    
# main()