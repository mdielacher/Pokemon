import numpy as np
from keras import layers, models
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
        model.add(layers.Dense(256, activation='relu', input_dim=self.noise_dim))  # Increased from 50 to 256
        model.add(layers.BatchNormalization(momentum=0.8))  # Add batch normalization
        model.add(layers.Dense(512, activation='relu'))  # New layer
        model.add(layers.BatchNormalization(momentum=0.8))  # Add batch normalization
        model.add(layers.Dense(1024, activation='relu'))  # New layer
        model.add(layers.BatchNormalization(momentum=0.8))  # Add batch normalization
        model.add(layers.Dense(self.data.shape[1], activation='tanh'))  # Changed to tanh activation
        return model
    
    
    def create_discriminator(self) -> models.Sequential:
        """Generate discriminator"""
        model = models.Sequential(name="discriminator_model")
        model.add(layers.Dense(64, activation='relu', input_dim=self.data.shape[1]))  # Reduced from 50 to 64
        model.add(layers.Dropout(0.4))  # Add dropout for regularization
        model.add(layers.Dense(128, activation='relu'))  # Reduced from 100 to 128
        model.add(layers.Dropout(0.4))  # Add dropout for regularization
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    def compile(self, generator, discriminator) -> models.Sequential:
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
            
            # Train the discriminator
            generated_data = generator.predict(noise)
            generated_data = generated_data.astype(np.float32)

            idx = np.random.randint(0, self.data.shape[0], self.batch_size)
            real_data = self.data.iloc[idx]

            # Prepare real and synthetic labels
            real_labels = np.ones((self.batch_size, 1), dtype=np.float32)
            fake_labels = np.zeros((self.batch_size, 1), dtype=np.float32)
            labels = np.concatenate([real_labels, fake_labels])
            X = np.concatenate([real_data, generated_data])
            discriminator.trainable = True
            d_loss , _ = discriminator.train_on_batch(X, labels)
            discriminator_losses.append(d_loss)

            # Train the generator
            noise = np.random.normal(0, 1, (self.batch_size, self.noise_dim))
            g_loss = gan.train_on_batch(noise, np.ones(self.batch_size))
            generator_losses.append(g_loss)


            print('>%d, d_loss=%.3f, g_loss=%.3f' %(epoch+1, d_loss, g_loss))

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        models.save_model(gan, filepath=f'models/evaluation/models/gan_model_{timestamp}.keras')
        keras.Model.save_weights(gan, f'models/evaluation/weights/gan_model_{timestamp}.weights.h5')
        

        return generator, discriminator_losses, generator_losses
    
    #TODO: generator evaluerien
    
    # def evaluate_discriminator(self, generator, discriminator):
    #     # Generate fake data
    #     generated_data = generator.predict(self.noise)
    #     generated_data = generated_data.astype(np.float32)

    #     # Real data
    #     real_data = self.data

    #     # Labels for real and fake data
    #     real_labels = np.ones((real_data.shape[0], 1), dtype=np.float32)
    #     fake_labels = np.zeros((generated_data.shape[0], 1), dtype=np.float32)

    #     # Evaluate discriminator on real data
    #     real_loss, real_accuracy = discriminator.evaluate(real_data, real_labels, verbose=0)

    #     # Evaluate discriminator on fake data
    #     fake_loss, fake_accuracy = discriminator.evaluate(generated_data, fake_labels, verbose=0)

    #     print(f'Real Data Accuracy: {real_accuracy*100:.2f}%')
    #     print(f'Fake Data Accuracy: {fake_accuracy*100:.2f}%')
    #     print(f'Overall Accuracy: {(real_accuracy + fake_accuracy) * 50:.2f}%')  # Average of real and fake accuracy

    
        

    

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