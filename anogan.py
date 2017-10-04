from __future__ import print_function

import os

from collections import defaultdict
# try:
#     import cPickle as pickle
# except ImportError:
#     import pickle
from PIL import Image

from six.moves import range

import matplotlib
import matplotlib.pyplot as plt

import datetime

import numpy as np
from scipy.stats import truncnorm

import keras.backend as K

from keras.models import Model

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import Activation, ZeroPadding2D

from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D

from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

from keras.regularizers import l2
from keras.optimizers import Adam

from keras.utils import plot_model
from keras.utils.generic_utils import Progbar

from sklearn.metrics import confusion_matrix

K.set_image_data_format('channels_last')


class Generator(object):
    """docstring for Generator."""

    def __init__(self, latent_shape, n_filters, filter_size, strides, batch_norm_momentum=0.8):

        # save initial args
        self.latent_shape        = latent_shape
        self.n_filters           = n_filters
        self.filter_size         = filter_size
        self.strides             = strides
        self.batch_norm_momentum = batch_norm_momentum

        # generator input of latent space vector Z, typically a 1D vector
        gen_input = Input(shape=self.latent_shape, name='generator_input')

        # layer 1 - high dimensional dense layer
        cnn = Dense(512 * 4 * 4)(gen_input)
        cnn = BatchNormalization(momentum=self.batch_norm_momentum)(cnn)
        cnn = LeakyReLU(alpha=0.2)(cnn)
        cnn = Reshape((4, 4, 512))(cnn)

        # layer 2-5 - high dimensional dense layer
        for nb_filters in self.n_filters:
            cnn = self.deconv2Dlayer(cnn, nb_filters)

            # layer 6 - Final convolution layer
        cnn = Conv2D(1, kernel_size=self.filter_size, padding='same')(cnn)
        gen_output = Activation('tanh', name='generator_output')(cnn)

        # Model definition with Functional API
        self.model = Model(gen_input, gen_output)

    def deconv2Dlayer(self, x, nb_filters):
        # Simpe Conv2DTranspose
        # Not good, compared to upsample + conv2d below.
        '''
        x = Conv2DTranspose(
            filters     = nb_filters,
            kernel_size = self.filter_size,
            strides     = self.strides,
            padding     = 'same'
        )(x)
        '''

        # Bilinear2x... Not sure if it is without bug, not tested yet.
        # Tend to make output blurry though
        # x = bilinear2x( x, filters )
        # x = Conv2D( filters, shape, padding='same' )( x )

        # simple and works
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(
            filters=nb_filters,
            kernel_size=self.filter_size,
            padding='same'
        )(x)

        x = BatchNormalization(momentum=self.batch_norm_momentum)(x)
        x = LeakyReLU(alpha=0.2)(x)
        return x

    def sample_z(self, batch_size, latent_shape):
        mu, sigma = 0, 0.4
        lower, upper = -1, 1

        return truncnorm.rvs(
            (lower - mu) / sigma, (upper - mu) / sigma,
            loc=mu,
            scale=sigma,
            size=(batch_size, self.latent_shape[0])
        )

    def generate_samples(self, batch_size=128, verbose=0):
        # We generate a "batch" of noise
        # noise = np.random.uniform(-1, 1, size=(batch_size, self.latent_shape[0]))

        # We prefer gaussian noise
        noise = self.sample_z(batch_size, self.latent_shape[0])

        # noise = np.random.normal(0, 1, size=(batch_size, self.latent_shape[0]))

        # We generate a batch of images with G
        return self.model.predict(noise, verbose=verbose)


class Discriminator(object):
    """docstring for Discriminator."""

    def __init__(self, input_shape, n_filters, filter_size, strides, batch_norm_momentum=0.6):

        # save initial args
        self.input_shape         = input_shape
        self.n_filters           = n_filters
        self.filter_size         = filter_size
        self.strides             = strides
        self.batch_norm_momentum = batch_norm_momentum

        # discriminator input of size related to image size.
        disc_input = Input(shape=self.input_shape, name='discriminator_input')

        # Layer 2-5 - Conv Layers
        for i, nb_filters in enumerate(self.n_filters[::-1]):  # List in reverse order
            if (i == 0):
                cnn = self.conv2Dlayer(disc_input, nb_filters)
            else:
                cnn = self.conv2Dlayer(cnn, nb_filters)

        # transform 3D Matrix to a 1D Vector
        cnn = Flatten()(cnn)

        # Layer 6 - Output Layer
        cnn = Dense(1, kernel_regularizer = l2(.01))(cnn)
        disc_output = Activation('sigmoid', name='discriminator_output')(cnn)

        # Model definition with Functional API
        self.model = Model(disc_input, disc_output)

    def conv2Dlayer(self, x, nb_filters):

        x = Conv2D(
            filters=nb_filters,
            kernel_size=self.filter_size,
            strides=self.strides,
            padding="same",
            kernel_regularizer=l2(.01)
        )(x)

        x = Dropout(0.5)(x)
        x = BatchNormalization(momentum=self.batch_norm_momentum)(x)
        x = LeakyReLU(alpha=0.2)(x)

        return x


class AnoGAN(object):
    """docstring for AnoGan."""

    def __init__(
            self,
            input_shape=(64, 64, 1),
            latent_shape=(100,),
            n_filters=[512, 256, 128, 64],
            filter_size=5,
            strides=2,
            gen_lr=2e-4,
            gen_beta1=0.5,
            dis_lr=2e-4,
            dis_beta1=0.5,
            model_name="",
            output_dirname="output",
            stats_step_interval=100
    ):

        # save initial args - Network Settings
        self.input_shape = input_shape
        self.latent_shape = latent_shape
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.strides = strides

        # save initial args - Optimizer Settings
        self.gen_lr = gen_lr
        self.gen_beta1 = gen_beta1

        self.dis_lr = dis_lr
        self.dis_beta1 = dis_beta1

        # save initial args - Training Process Settings
        self.model_name = model_name
        self.outdir = os.path.join(output_dirname, self.model_name)
        self.stats_step_interval = stats_step_interval

        print("Model Name: %s" % self.model_name)

        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        # ================= Optimitizers ================= #

        # We create the optimizer for D
        # Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        d_optim = Adam(lr=self.dis_lr, beta_1=self.dis_beta1)

        # We create the optimizer for G
        # Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        g_optim = Adam(lr=self.gen_lr, beta_1=self.gen_beta1)

        # ======= Models Creation and Compilation ======== #

        # create the discriminator
        self.discriminator = Discriminator(
            input_shape=self.input_shape,
            n_filters=self.n_filters,
            filter_size=self.filter_size,
            strides=self.strides
        )
        self.discriminator.model.compile(loss='binary_crossentropy', optimizer=d_optim, metrics=['accuracy'])

        # create the generator
        self.generator = Generator(
            latent_shape=self.latent_shape,
            n_filters=self.n_filters,
            filter_size=self.filter_size,
            strides=self.strides
        )
        self.generator.model.compile(loss='binary_crossentropy', optimizer=g_optim)

        # create an Input Layer for the complete GAN
        gan_input = Input(shape=self.latent_shape)

        # link the input of the GAN to the Generator
        G_output = self.generator.model(gan_input)

        # For the combined model we will only train the generator => We do not want to backpropagate D while training G
        self.discriminator.model.trainable = False

        # we retrieve the output of the GAN
        gan_output = self.discriminator.model(G_output)

        # we construct a model out of it.
        self.fullgan_model = Model(gan_input, gan_output)
        self.fullgan_model.compile(loss='binary_crossentropy', optimizer=g_optim, metrics=['accuracy'])

    def generator_summary(self):
        plot_model(
            self.generator.model,
            to_file=os.path.join(self.outdir, 'generator.png'),
            show_shapes=True,
            show_layer_names=True
        )
        self.generator.model.summary()

    def discriminator_summary(self):
        self.discriminator.model.trainable = True

        plot_model(
            self.discriminator.model,
            to_file=os.path.join(self.outdir, 'discriminator.png'),
            show_shapes=True,
            show_layer_names=True
        )
        self.discriminator.model.summary()

        self.discriminator.model.trainable = False

    def full_GAN_summary(self):
        self.discriminator.model.trainable = True
        plot_model(
            self.fullgan_model,
            to_file=os.path.join(self.outdir, 'full_GAN.png'),
            show_shapes=True,
            show_layer_names=True
        )
        self.fullgan_model.summary()

        self.discriminator.model.trainable = False

    def _epoch_stats_writer(
            self,
            start_timer,
            step_num,
            epoch_num,
            epoch_gen_loss,
            epoch_disc_loss,
            epoch_gen_acc,
            epoch_disc_acc,
            X_val_generator=None,
            last_epoch=False
    ):

        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)
        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)
        generator_train_acc = np.mean(np.array(epoch_gen_acc), axis=0)
        discriminator_train_acc = np.mean(np.array(epoch_disc_acc), axis=0)

        # generate an epoch report on performance
        self.train_history['generator']["loss"].append(generator_train_loss)
        self.train_history['generator']["accuracy"].append(generator_train_acc)

        self.train_history['discriminator']["loss"].append(discriminator_train_loss)
        self.train_history['discriminator']["accuracy"].append(discriminator_train_acc)

        # =============================================
        # ============= Print Statistics ==============
        # =============================================

        end_timer = datetime.datetime.now()

        print("\nEpoch processed in %ds" % int((end_timer - start_timer).total_seconds()))

        stats_dict = {
            'generator': {
                'loss': self.train_history['generator']["loss"][-1],
                'accuracy': self.train_history['generator']["accuracy"][-1]
            },
            'discriminator': {
                'loss': self.train_history['discriminator']["loss"][-1],
                'accuracy': self.train_history['discriminator']["accuracy"][-1]
            }
        }

        self._stats_writer(
            step_num=step_num,
            epoch_num=epoch_num,
            stats_dict=stats_dict,
            X_val_generator=X_val_generator,
            is_epoch=True
        )

        # =============================================
        # =========== Plot Final Statistics ===========
        # =============================================

        if last_epoch:
            # ==== print a plot of the accuracy of D ==== #

            fig = plt.figure()

            plt.plot(self.train_history['discriminator']["accuracy"], label='discriminator accuracy on real data')
            plt.plot(self.train_history['generator']["accuracy"], label='discriminator accuracy on fake data')
            plt.axhline(y=.5, color='#d62728', label='nash equilibrium')

            fig.suptitle("Discriminator's accuracy during training", fontsize=20)
            plt.xlabel('Training Steps', fontsize=18)
            plt.ylabel('Accuracy', fontsize=16)

            plt.axis([0, len(self.train_history['discriminator']["accuracy"]), 0, 1])
            plt.legend()

            plt.show()

            # == print a plot of the costs for G and D == #

            fig = plt.figure()

            plt.plot(self.train_history['discriminator']["loss"], label='discriminator cost')
            plt.plot(self.train_history['generator']["loss"], label='generator cost')

            fig.suptitle("D and G training cost over time", fontsize=20)
            plt.xlabel('Training Steps', fontsize=18)
            plt.ylabel('Training Cost', fontsize=16)

            max_height = max(self.train_history['discriminator']["loss"] + self.train_history['generator']["loss"])
            min_height = min(self.train_history['discriminator']["loss"] + self.train_history['generator']["loss"])

            plt.axis([0, len(self.train_history['discriminator']["loss"]), min_height, max_height])
            plt.legend()

            plt.show()

    def _step_stats_writer(
            self,
            step_num,
            epoch_num,
            epoch_gen_loss,
            epoch_disc_loss,
            epoch_gen_acc,
            epoch_disc_acc
    ):

        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)
        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)
        generator_train_acc = np.mean(np.array(epoch_gen_acc), axis=0)
        discriminator_train_acc = np.mean(np.array(epoch_disc_acc), axis=0)

        stats_dict = {
            'generator': {
                'loss': generator_train_loss,
                'accuracy': generator_train_acc
            },
            'discriminator': {
                'loss': discriminator_train_loss,
                'accuracy': discriminator_train_acc
            }
        }

        print()  # Add a linebreak

        self._stats_writer(
            step_num=step_num,
            epoch_num=epoch_num,
            stats_dict=stats_dict
        )

    def _stats_writer(
            self,
            step_num,
            epoch_num,
            stats_dict,
            X_val_generator=None,
            is_epoch=False
    ):

        #################################################
        # ==== Print statistics to follow learning ==== #
        #################################################

        print('\n%s | %s | %s | %s | %s | %s | %s |' % (
            'network'.ljust(35),
            'loss'.ljust(10),
            'accuracy'.ljust(10),
            'recall'.ljust(7),
            'precision'.ljust(10),
            'specificity'.ljust(13),
            'NPV'.ljust(6)
        )
              )
        print('-' * 110)
        print('%s | %s | %s | %s | %s | %s | %s |' % (
            'generator (train)'.ljust(35),
            ('%.3f' % stats_dict['generator']["loss"]).ljust(10),
            ('%.3f' % stats_dict['generator']["accuracy"]).ljust(10),
            ''.ljust(7),
            ''.ljust(10),
            ''.ljust(13),
            ''.ljust(6)
        ))
        print('%s | %s | %s | %s | %s | %s | %s |' % (
            'discriminator (train)'.ljust(35),
            ('%.3f' % stats_dict['discriminator']["loss"]).ljust(10),
            ('%.3f' % stats_dict['discriminator']["accuracy"]).ljust(10),
            ''.ljust(7),
            ''.ljust(10),
            ''.ljust(13),
            ''.ljust(6)
        ))

        if (not X_val_generator is None):
            precision, recall, specificity, NPV = self.detect_defects(X_val_generator, verbose=0)

            print('%s | %s | %s | %s | %s | %s | %s |' % (
                'detection rate (validation)'.ljust(35),
                ''.ljust(10),
                ''.ljust(10),
                ('%.3f' % recall).ljust(7),
                ('%.3f' % precision).ljust(10),
                ('%.3f' % specificity).ljust(13),
                ('%.3f' % NPV).ljust(6)
            ))

        if (is_epoch):
            print('\n' + ('#' * 115) + '\n')
        else:
            print('\n' + ('=' * 110) + '\n')

        ################################################
        # == Generate and Save a few samples from G == #
        ################################################

        generated_images = self.generator.generate_samples(batch_size=100, verbose=0)

        # arrange generated data into a grid
        img = (np.concatenate([r.reshape(-1, 64)
                               for r in np.split(generated_images, 10)
                               ], axis=-1) * 127.5 + 127.5).astype(np.uint8)

        filename = 'AnoGAN_epoch_{epoch_num:0>2}_step_{step_num:0>4}.png'.format(
            epoch_num=epoch_num,
            step_num=step_num
        )

        Image.fromarray(img).save(os.path.join(self.outdir, filename))

    def _training_step(self, X, d_acc_threshold=0.35, verbose=0):

        # =============================================
        # ============ Generator Training =============
        # =============================================

        self.discriminator.model.trainable = False
        for l in self.discriminator.model.layers: l.trainable = False

        g_loss = 0.0
        g_acc  = 0.0

        # We train G twice, because D is trained twice at each step
        for _ in range(2):
            noise = self.generator.sample_z(self.batch_size, self.latent_shape[0])

            loss, acc = self.fullgan_model.train_on_batch(noise, np.ones(noise.shape[0]))

            g_loss = np.add(g_loss, loss)
            g_acc  = np.add(g_acc, acc)

        g_loss *= 0.5
        g_acc  *= 0.5

        # =============================================
        # ========== Discriminator Training ===========
        # =============================================

        if g_acc >= d_acc_threshold:  # Preventing D to become too strong

            self.discriminator.model.trainable = True
            for l in self.discriminator.model.layers: l.trainable = True

            # We generate a batch of images with G
            generated_images = self.generator.generate_samples(batch_size=self.batch_size, verbose=verbose)

            d_loss_fake, d_acc_fake = self.discriminator.model.train_on_batch(
                generated_images,
                np.zeros(generated_images.shape[0])
            )
            d_loss_real, d_acc_real = self.discriminator.model.train_on_batch(X, np.ones(X.shape[0]))

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            d_acc  = 0.5 * np.add(d_acc_real,  d_acc_fake)

        else:
            d_loss = None
            d_acc  = None

        return g_loss, g_acc, d_loss, d_acc

    def train_generator(self, X_train_generator, epochs, X_val_generator=None, verbose=0):

        # save initial args
        self.epochs = epochs
        self.batch_size = X_train_generator.batch_size

        # variables initialization
        self.train_history = dict()
        self.train_history['generator'] = defaultdict(list)
        self.train_history['discriminator'] = defaultdict(list)

        # Create a progress bar
        n_steps = int(X_train_generator.samples / self.batch_size)

        for epoch in range(self.epochs):
            progress_bar = Progbar(target=n_steps)

            print('Epoch {} of {}\n'.format(epoch + 1, epochs))

            tic = datetime.datetime.now()

            last_stats_idx_gen  = 0
            last_stats_idx_disc = 0

            epoch_gen_loss = list()
            epoch_disc_loss = list()

            epoch_gen_acc = list()
            epoch_disc_acc = list()

            for step in range(n_steps):

                #############################################
                # ========= Launch Model Training ========= #
                #############################################

                # We update the progress bar
                progress_bar.update(step + 1, force=True)  # step needs to be increment because it starts at 0.

                # rescaling images: pixel_values in [-1, 1]
                image_batch = (X_train_generator.next().astype(np.float32) - 127.5) / 127.5

                g_loss, g_acc, d_loss, d_acc = self._training_step(image_batch, verbose)

                if not d_loss is None:
                    epoch_disc_loss.append(d_loss)
                    epoch_disc_acc.append(d_acc)

                epoch_gen_loss.append(g_loss)
                epoch_gen_acc.append(g_acc)

                #############################################
                # ======= Write intermediate Stats ======== #
                #############################################

                if ((step + 1) % self.stats_step_interval == 0 or ((step + 1) == n_steps)):

                    if ((step + 1) == n_steps):

                        if ((epoch + 1) == self.epochs):
                            last_epoch = True
                        else:
                            last_epoch = False

                        self._epoch_stats_writer(
                            start_timer=tic,
                            step_num=n_steps,
                            epoch_num=epoch + 1,
                            epoch_gen_loss=epoch_gen_loss,
                            epoch_disc_loss=epoch_disc_loss,
                            epoch_gen_acc=epoch_gen_acc,
                            epoch_disc_acc=epoch_disc_acc,
                            X_val_generator=X_val_generator,
                            last_epoch=last_epoch
                        )

                    else:
                        self._step_stats_writer(
                            step_num=step + 1,
                            epoch_num=epoch + 1,
                            epoch_gen_loss=epoch_gen_loss[last_stats_idx_gen:],
                            epoch_disc_loss=epoch_disc_loss[last_stats_idx_disc:],
                            epoch_gen_acc=epoch_gen_acc,
                            epoch_disc_acc=epoch_disc_acc
                        )

                        last_stats_idx_gen  = len(epoch_gen_loss)
                        last_stats_idx_disc = len(epoch_disc_loss)

    def detect_defects(self, validation_generator, verbose=1):

        total_samples = validation_generator.samples
        batch_size = validation_generator.batch_size

        results = list()
        labels = list()

        if (verbose != 0):
            progress_bar = Progbar(target=total_samples)

        for _ in range(np.ceil(total_samples / batch_size).astype(np.int32)):

            image_batch, lbls = validation_generator.next()

            labels = np.append(labels, lbls.reshape(lbls.shape[0]))
            image_batch = (image_batch.astype(np.float32) - 127.5) / 127.5

            tmp_rslt = self.discriminator.model.predict(
                x=image_batch,
                batch_size=image_batch.shape[0],
                verbose=0
            )

            if (verbose != 0):
                progress_bar.add(image_batch.shape[0])

            results = np.append(results, tmp_rslt.reshape(tmp_rslt.shape[0]))

        results = [1 if x >= 0.5 else 0 for x in results]

        tn, fp, fn, tp = confusion_matrix(labels, results).ravel()

        #################### NON DEFECT SITUATIONS ####################

        # Probability of Detecting a Non-Defect: (tp / (tp + fn))
        if ((tp + fn) != 0):
            recall = tp / (tp + fn)
        else:
            recall = 0.0

        # Probability of Correctly Detecting a Non-Defect: (tp / (tp + fp))

        if ((tp + fp) != 0):
            precision = tp / (tp + fp)
        else:
            precision = 0.0

        ###################### DEFECT SITUATIONS ######################

        # Probability of Detecting a Defect: (tn / (tn + fp))
        if ((tn + fp) != 0):
            specificity = tn / (tn + fp)
        else:
            specificity = 0.0

        # Probability of Correctly Detecting a Defect: (tn / (tn + fn))
        if ((tn + fn) != 0):
            negative_predictive_value = tn / (tn + fn)
        else:
            negative_predictive_value = 0.0

        return precision, recall, specificity, negative_predictive_value
