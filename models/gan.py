import numpy as np
from keras import layers, models, initializers
from datetime import datetime
import keras

class GAN():
    """Class to train a GAN model"""

    def __init__(self, data, noise_dim, epochs, batch_size) -> None:
        self.data = data
        self.noise_dim = noise_dim
        self.epochs = epochs
        self.batch_size = batch_size

    def create_generator(self) -> models.Sequential:
        """Generate generator"""
        model = models.Sequential(name="generator_model")

        model.add(layers.Dense(64, input_dim=self.noise_dim))  
        model.add(layers.Dropout(0.2))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.BatchNormalization(momentum=0.8))  

        model.add(layers.Dense(128))  
        model.add(layers.Dropout(0.2))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.BatchNormalization(momentum=0.8))

        model.add(layers.Dense(256)) 
        model.add(layers.Dropout(0.2))
        model.add(layers.LeakyReLU(alpha=0.2)) 
        model.add(layers.BatchNormalization(momentum=0.8))  
        model.add(layers.Dense(self.data.shape[1], activation='tanh'))  
        return model
    
    
    def create_discriminator(self) -> models.Sequential:
        """Generate discriminator"""
        init = initializers.RandomNormal(mean=0.0, stddev=0.02)
        model = models.Sequential(name="discriminator_model")

        model.add(layers.Dense(256,  input_dim=self.data.shape[1], kernel_initializer=init)) 
        model.add(layers.LeakyReLU(alpha=0.2))

        model.add(layers.Dense(128, activation='relu', kernel_initializer=init)) 
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.4))

        model.add(layers.Dense(64, kernel_initializer=init))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.4))

        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    def model_compile(self, generator, discriminator) -> models.Sequential:
        """Combine discriminator and generator to one model

        Args:
            generator (_type_): Generator Model
            discriminator (_type_): Discriminator Model

        Returns:
            models.Sequential: GAN
        """
        discriminator.trainable = False
        generator.trainable = True
        model = models.Sequential(name="GAN")
        model.add(generator)
        model.add(discriminator)
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model

    def train(self, generator, discriminator, gan):
        discriminator_losses = []
        generator_losses = []
        for epoch in range(self.epochs):
            noise = np.random.normal(0, 1, (self.batch_size, self.noise_dim))
            
            generated_data = generator.predict(noise)
            generated_data = generated_data.astype(np.float32)

            idx = np.random.randint(0, self.data.shape[0], self.batch_size)
            real_data = self.data.iloc[idx]

            real_labels = np.ones((self.batch_size, 1), dtype=np.float32)
            fake_labels = np.zeros((self.batch_size, 1), dtype=np.float32)
            labels = np.concatenate([real_labels, fake_labels])
            X = np.concatenate([real_data, generated_data])
            discriminator.trainable = True
            d_loss , _ = discriminator.train_on_batch(X, labels)
            discriminator_losses.append(d_loss)

            noise = np.random.normal(0, 1, (self.batch_size, self.noise_dim))
            g_loss = gan.train_on_batch(noise, np.ones(self.batch_size))
            generator_losses.append(g_loss)


            print('>%d, d_loss=%.3f, g_loss=%.3f' %(epoch+1, d_loss, g_loss))

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        models.save_model(gan, filepath=f'models/evaluation/models/gan_model_{timestamp}.keras')
        keras.Model.save_weights(gan, f'models/evaluation/weights/gan_model_{timestamp}.weights.h5')
        

        return generator, discriminator_losses, generator_losses
    