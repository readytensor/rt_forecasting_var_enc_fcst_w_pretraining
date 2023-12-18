import os
import sys
import warnings
from typing import List, Union
import math

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.optimizers import Adam

# from prediction.vae_dense_model import VariationalAutoencoderDense as VAE
from prediction.vae_conv_model import VariationalAutoencoderConv as VAE
from prediction.pretraining_data_gen import get_pretraining_data


# Check for GPU availability
IS_GPU_AVAI = (
    "GPU available (YES)"
    if tf.config.list_physical_devices("GPU")
    else "GPU not available"
)
print(IS_GPU_AVAI)

PREDICTOR_FILE_NAME = "predictor.joblib"
MODEL_PARAMS_FNAME = "model_params.save"
MODEL_ENCODER_WTS_FNAME = "model_encoder_wts.save"
MODEL_DECODER_WTS_FNAME = "model_decoder_wts.save"
HISTORY_FNAME = "history.json"
COST_THRESHOLD = float("inf")

class InfCostStopCallback(Callback):
    """Callback to check if cost has hit infinity. Then we stop training."""
    def on_epoch_end(self, epoch, logs={}):
        loss_val = logs.get('loss')
        if(loss_val == COST_THRESHOLD or tf.math.is_nan(loss_val)):
            print("Cost is inf, so stopping training!!")
            self.model.stop_training = True

def get_patience_factor(N):
    # magic number - just picked through trial and error
    patience = max(4, int(38 - math.log(N, 1.5)))
    return patience


class Forecaster:
    """A wrapper class for the Variational Encoding Forecaster.

    This class provides a consistent interface that can be used with other
    Forecaster models.
    """
    MIN_HISTORY_LEN = 60        # in epochs
    MODEL_NAME = "Variational_Encoding_Forecaster"

    def __init__(
            self,
                encode_len,
                decode_len,
                num_exog,
                latent_dim,
                first_hidden_dim,
                second_hidden_dim,
                reconstruction_wt=5.0, **kwargs ):
        
        self.encode_len = encode_len
        self.decode_len = decode_len
        self.feat_dim = 1+num_exog
        self.latent_dim = latent_dim
        self.first_hidden_dim = first_hidden_dim
        self.second_hidden_dim = second_hidden_dim
        self.hidden_layer_sizes = [int(first_hidden_dim), int(second_hidden_dim)]
        self.reconstruction_wt = reconstruction_wt
        self.batch_size = 32

        self.vae_model = VAE(
            encode_len=encode_len,
            decode_len=decode_len,
            feat_dim=self.feat_dim,
            latent_dim = latent_dim,
            hidden_layer_sizes=self.hidden_layer_sizes,
            reconstruction_wt = reconstruction_wt
        )   
        self.learning_rate = 1e-4
        self.vae_model.compile(optimizer=Adam(self.learning_rate))
        self._is_trained = False
        # self.vae_model.encoder.summary()
        # self.vae_model.decoder.summary()

    def _get_X_and_y(self, data: np.ndarray, is_train:bool=True) -> np.ndarray:
        """Extract X (historical target series) and y (forecast window target) 
           from given array of shape [N, T, D]

            When is_train is True, data contains both history and forecast windows.
            When False, only history is contained.
        """
        N, T, D = data.shape
        if D != self.feat_dim:
            raise ValueError(
                f"Training data expected to have {self.feat_dim-1} exogenous variables. "
                f"Found {D-1}"
            )
        if is_train:
            if T != self.encode_len + self.decode_len:
                raise ValueError(
                    f"Training data expected to have {self.encode_len + self.decode_len}"
                    f" length on axis 1. Found length {T}"
                )
            X = data[:, :self.encode_len, :]
            y = data[:, self.encode_len:, 0]
        else:
            # for inference
            if T < self.encode_len:
                raise ValueError(
                    f"Inference data length expected to be >= {self.encode_len}"
                    f" on axis 1. Found length {T}"
                )
            X = data[:, -self.encode_len:, :]
            y = None
        return X, y

    def _train_on_data(self, data, validation_split=0.1, verbose=1, max_epochs=500):
        """Train the model on the given data.

        Args:
            data (pandas.DataFrame): The training data.
        """
        X, y = self._get_X_and_y(data, is_train=True)
        print("X and y shapes:", X.shape, y.shape)
        loss_to_monitor = 'loss' if validation_split is None else 'val_loss'
        patience = get_patience_factor(X.shape[0])
        early_stop_callback = EarlyStopping(
            monitor=loss_to_monitor, min_delta = 1e-4, patience=patience)
        learning_rate_reduction = ReduceLROnPlateau(
            monitor=loss_to_monitor,
            patience=patience//2,
            factor=0.5,
            min_lr=1e-7
        )
        if X.shape[0] < 100:
            validation_split = None
        history = self.vae_model.fit(
            x=X,
            y=y,
            validation_split=validation_split,
            verbose=verbose,
            epochs=max_epochs,
            callbacks=[early_stop_callback, learning_rate_reduction],
            batch_size=self.batch_size,
            shuffle=True
        )
        # recompile the model to reset the optimizer; otherwise re-training slows down
        self.vae_model.compile(optimizer=Adam(self.learning_rate))
        return history

    def fit(self, training_data:np.ndarray, pre_training_data: Union[np.ndarray, None]=None,
            validation_split: Union[float, None]=0.1, verbose:int=1,
            max_epochs:int=1000):

        """Fit the Forecaster to the training data.
        A separate Prophet model is fit to each series that is contained
        in the data.

        Args:
            data (pandas.DataFrame): The features of the training data.
        """
        if pre_training_data is not None:
            print("Conducting pretraining...")
            _ = self._train_on_data(
                data=pre_training_data,
                validation_split=validation_split,
                verbose=verbose,
                max_epochs=max_epochs
            )
        
        print("Training on main data...")
        history = self._train_on_data(
            data=training_data,
            validation_split=validation_split,
            verbose=verbose,
            max_epochs=max_epochs
        )
        self._is_trained = True
        return history

    def predict(self, test_data: np.ndarray) -> np.ndarray:
        """Make the forecast of given length.

        Args:
            test_data (np.ndarray): Given test input for forecasting.
        Returns:
            numpy.ndarray: predictions as numpy array.
        """
        X, y = self._get_X_and_y(test_data, is_train=False)
        preds = self.vae_model.predict(x=X)
        return np.expand_dims(preds, axis=-1)
    
    def evaluate(self, test_data: np.ndarray) -> np.ndarray:
        """Return loss for given evaluation X and y

        Args:
            test_data (np.ndarray): Given test data for evaluation.
        Returns:
            float: loss value (mse).
        """
        X, y = self._get_X_and_y(test_data, is_train=False)
        score = self.vae_model.evaluate(x=X, y=y)
        return score

    def save(self, model_dir_path: str) -> None:
        """Save the forecaster to disk.

        Args:
            model_dir_path (str): The dir path to which to save the model.
        """
        if self.vae_model is None:
            raise NotFittedError("Model is not fitted yet.")
        encoder_wts = self.vae_model.encoder.get_weights()
        decoder_wts = self.vae_model.decoder.get_weights()
        joblib.dump(encoder_wts, os.path.join(model_dir_path, MODEL_ENCODER_WTS_FNAME))
        joblib.dump(decoder_wts, os.path.join(model_dir_path, MODEL_DECODER_WTS_FNAME))
        model_params = {
            'encode_len': self.encode_len,
            'decode_len': self.decode_len,
            'feat_dim': self.feat_dim,
            'latent_dim': self.latent_dim,
            'first_hidden_dim': self.first_hidden_dim,
            'second_hidden_dim': self.second_hidden_dim,
            'reconstruction_wt': self.reconstruction_wt,
        }
        joblib.dump(model_params, os.path.join(model_dir_path, MODEL_PARAMS_FNAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Forecaster":
        """Load the forecaster from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Forecaster: A new instance of the loaded forecaster.
        """
        dict_params = joblib.load(os.path.join(model_dir_path, MODEL_PARAMS_FNAME))
        model = cls(
            encode_len = dict_params['encode_len'],
            decode_len = dict_params['decode_len'],
            num_exog = dict_params['feat_dim']-1,
            latent_dim = dict_params['latent_dim'],
            first_hidden_dim=dict_params['first_hidden_dim'],
            second_hidden_dim=dict_params['second_hidden_dim'],
            reconstruction_wt = dict_params['reconstruction_wt']
        )

        first_hidden_dim = int(dict_params['first_hidden_dim'])
        second_hidden_dim = int(dict_params['second_hidden_dim'])

        model.vae_model = VAE(
            encode_len = dict_params['encode_len'],
            decode_len = dict_params['decode_len'],
            feat_dim = dict_params['feat_dim'],
            latent_dim = dict_params['latent_dim'],
            hidden_layer_sizes=[first_hidden_dim, second_hidden_dim ],
            reconstruction_wt = dict_params['reconstruction_wt'],
        )
        encoder_wts = joblib.load(os.path.join(model_dir_path, MODEL_ENCODER_WTS_FNAME))
        decoder_wts = joblib.load(os.path.join(model_dir_path, MODEL_DECODER_WTS_FNAME))
        model.vae_model.encoder.set_weights(encoder_wts)
        model.vae_model.decoder.set_weights(decoder_wts)
        model.vae_model.compile(optimizer=Adam(learning_rate = 1e-4))
        return model

    def __str__(self):
        return f"Model name: {self.MODEL_NAME}"


def train_predictor_model(
    history: pd.DataFrame,
    forecast_length: int,
    frequency: str,
    hyperparameters: dict,
) -> Forecaster:
    """
    Instantiate and train the forecaster model.

    Args:
        history (np.ndarray): The training data inputs.
        forecast_length (int): Length of forecast window.
        frequency (str): Frequency of the data such as MONTHLY, DAILY, etc.
        hyperparameters (dict): Hyperparameters for the Forecaster.

    Returns:
        'Forecaster': The Forecaster model
    """
    pre_training_data = get_pretraining_data(
        series_len=history.shape[1],
        forecast_length=forecast_length,
        frequency=frequency,
        num_exog=history.shape[2]-1
    )
    model = Forecaster(
        encode_len=history.shape[1] - forecast_length,
        decode_len=forecast_length,
        num_exog=history.shape[2] - 1,
        **hyperparameters,
    )
    model.fit(
        training_data=history,
        pre_training_data=pre_training_data,
    )
    return model


def predict_with_model(
    model: Forecaster, test_data: np.ndarray
) -> np.ndarray:
    """
    Make forecast.

    Args:
        model (Forecaster): The Forecaster model.
        test_data (np.ndarray): The test input data for forecasting.

    Returns:
        np.ndarray: The forecast.
    """
    return model.predict(test_data)


def save_predictor_model(model: Forecaster, predictor_dir_path: str) -> None:
    """
    Save the Forecaster model to disk.

    Args:
        model (Forecaster): The Forecaster model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Forecaster:
    """
    Load the Forecaster model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Forecaster: A new instance of the loaded Forecaster model.
    """
    return Forecaster.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Forecaster, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the Forecaster model and return the accuracy.

    Args:
        model (Forecaster): The Forecaster model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The labels of the test data.

    Returns:
        float: The accuracy of the Forecaster model.
    """
    return model.evaluate(x_test, y_test)
