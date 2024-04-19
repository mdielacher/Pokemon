import numpy as np
from keras import layers, models


class GAN():
    """Class to train a GAN model"""

    def __init__(self, data, noise, epochs) -> None:
        self.data = data
        self.noise = noise
        self.epochs = epochs

    def create_generator(self) -> models.Sequential:
        """Generate generator"""
        model = models.Sequential(name="generator_model")
        model.add(layers.Dense(15, activation='relu',input_dim=self.data.shape[1]))
        model.add(layers.Dense(30, activation='relu'))
        model.add(layers.Dense(self.data.shape[1], activation='sigmoid'))
        return model
    
    
    def create_discriminator(self) -> models.Sequential:
        """Generate discriminator"""
        model = models.Sequential(name="discriminator_model")
        model.add(layers.Dense(25, activation='relu',input_dim=self.data.shape[1]))
        model.add(layers.Dense(50, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
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
            print(len(X))
            print(len(labels))
            d_loss , _ = discriminator.train_on_batch(X, labels)

            # Train the generator
            g_loss = gan.train_on_batch(self.noise, np.ones(self.data.shape[0]))


            print('>%d, d1=%.3f, d2=%.3f' %(epoch+1, d_loss, g_loss))

        return generator
    
    #TODO: generator evaluerien
    
    def evaluate_discriminator(self, generator, discriminator):
        # Generate fake data
        generated_data = generator.predict(self.noise)
        generated_data = generated_data.astype(np.float32)

        # Real data
        real_data = self.data

        # Labels for real and fake data
        real_labels = np.ones((real_data.shape[0], 1), dtype=np.float32)
        fake_labels = np.zeros((generated_data.shape[0], 1), dtype=np.float32)

        # Evaluate discriminator on real data
        real_loss, real_accuracy = discriminator.evaluate(real_data, real_labels, verbose=0)

        # Evaluate discriminator on fake data
        fake_loss, fake_accuracy = discriminator.evaluate(generated_data, fake_labels, verbose=0)

        print(f'Real Data Accuracy: {real_accuracy*100:.2f}%')
        print(f'Fake Data Accuracy: {fake_accuracy*100:.2f}%')
        print(f'Overall Accuracy: {(real_accuracy + fake_accuracy) * 50:.2f}%')  # Average of real and fake accuracy

    
        

    

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