"""
	Some constant utilities that all the models will be using:
	1. The evaluation metric used for the competition is Mean Squared Logarithmic Error 
		hence all the models are going to be using the MSLE as their evaluation metric.
	2. The Default optimizer will be Adam and RSMprop, also we could choose to use a 
		learning_rate modifier for the Optimizer which would either be a InverseTimeDecay 
		or ExponentialDecay.
	3. They are only few type of layers and activation that work reasonably for Regression problems:
			Layers: Dense, Dropout, BatchNormalization
			activations: relu, elu, tanh*
	4. For each model, there should be a config dictionary that descrives values that best work for
		the model and this values should have been driven by testing and parameter tuning (potentially
		using Keras-tuner).
	5. Each function that implements the model gets a full configuration on both compilation and 
		fitting parameters that the model going to need.
	6. The monitored parameter in a EarlyStopping object is always 'val_loss', since the assumption
		is that we will be using validation_data in order to validate our data
	
	Note01: BatchNormalization layers should be used only at the beginning layer of the model since
		it does not work when used in different parts of the model.
	Note02: tanh does not happen to work as well as elu and relu, on a broader perspective relu is 
		most appropriate one for the regression.
"""


# Scheduler objects to control the optimizer learning rate:
from tensorflow.keras.optimizers.schedules import InverseTimeDecay, ExponentialDecay

def TimeDecayScheduler(learning_rate=0.001, decay_steps=200, decay_rate=1.2, name=""):
	""" Returns an InverseTimeDecay object with the given properties to be used in the optimizer. """
	return InverseTimeDecay(
		initial_learning_rate=learning_rate, 
		decay_steps=decay_steps,
		decay_rate=decay_rate,
		name=name
    )


def ExponentialScheduler(initial_learning_rate, decay_steps, decay_rate, name=""):
	""" Returns an ExponentialDecay object with the given properties to be used in the optimizer. """
	return InverseTimeDecay(
		initial_learning_rate=initial_learning_rate, 
		decay_steps=decay_steps,
		decay_rate=decay_rate,
		name=name
    )


# Actual Optimizers: Adam and RMSprop are the main two optimizers that are going to be used for this project since they accept schedulers and happen to be effective.

from tensorflow.keras.optimizers import Adam, RMSprop

def AdamOptimizer(learning_rate=0.001, scheduler=None):
	"""
		# params:
			learning_rate: the initial learning rate to be used
			scheduler: If this is passed by the user then use it in the optimizer instead of the learning rate
		
		# returns: an Adam optimizer
	"""
	if scheduler == None:
		return Adam(learning_rate)
	else:
		return Adam(scheduler)
	

def RMSpropOptimizer(learning_rate=0.001, scheduler=None):
	"""
		# params:
			learning_rate: the initial learning rate to be used
			scheduler: If this is passed by the user then use it in the 
						optimizer instead of the learning rate
		
		# returns: an RMSprop optimizer
	"""
	if scheduler == None:
		return RMSprop(learning_rate)
	else:
		return RMSprop(scheduler)

# CallBacks:
from tensorflow.keras.callbacks import EarlyStopping

def EarlyStopCallBack(patience=100):
	"""
		# params: patience of the object for the number of epochs passed with no improvement
		# returns: a EarlyStopping callback object 
	"""
	return EarlyStopping(monitor='val_loss', patience=patience)
	
	
# Models: 
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization  # Layers 
from tensorflow.keras.regularizers import l2, l1, l1_l2, L1L2  # Regularizer
from tensorflow.keras.losses import MeanSquaredLogarithmicError # Error-metric
import tensorflow_docs as tfdocs # For logging puposes


# This model still needs some modifiations
def NN01():
    """
        The architecture of this model consists of three batches of three Dense layers where:
            - The layer in between is linear and does not have an activation function, the other
                two layers have the relu activation function. The number of layer in between is four times
                the number of its sourounding layers.
            - The regularazation term for the middle layer is both higher and uses the l2 for its weights.
                There is more bias being introduced in the middle layer as well, the weights have a higher range
                in the midddle layer.
            - Added BatchNormalization layer after each three batches(testing phase).
            - The last three layers will be highly biased with high regualarization terms
    """
    model = keras.Sequential([
        InputLayer(input_shape=[len(X.keys())]),
        
        Dense(64, activation='elu', 
              kernel_regularizer=l1(0.001),
              bias_regularizer=l2(0.001),
              bias_initializer=TruncatedNormal(mean=0, stddev=0.005),
              kernel_initializer=TruncatedNormal(mean=0, stddev=25)
        ),
        Dense(256, 
              kernel_regularizer=l2(0.01),
              bias_regularizer=l1(0.01),
              bias_initializer=TruncatedNormal(mean=0, stddev=0.5),
              kernel_initializer=TruncatedNormal(mean=0, stddev=5)
        ),
        BatchNormalization(),
        Dense(64, activation='elu',
              bias_regularizer=l1(0.001), 
              kernel_regularizer=l2(0.001),
              bias_initializer=TruncatedNormal(mean=0, stddev=0.005),
              kernel_initializer=TruncatedNormal(mean=0, stddev=25)
        ),
        
        BatchNormalization(),
        Dense(128, activation = 'relu', 
              bias_regularizer=l1(0.001), 
              kernel_regularizer=l1(0.001),
              bias_initializer=TruncatedNormal(mean=0, stddev=1.5),
              kernel_initializer=TruncatedNormal(mean=0, stddev=1)
        ),
        Dense(512, 
              kernel_regularizer=l2(0.01),
              bias_regularizer=l1(0.01), 
              bias_initializer=TruncatedNormal(mean=0, stddev=1.5),
              kernel_initializer=TruncatedNormal(mean=0, stddev=5)
        ),
        BatchNormalization(),
        Dense(128, activation = 'elu',
              kernel_regularizer=l2(0.01),
              bias_regularizer=l1(0.001), 
              bias_initializer=TruncatedNormal(mean=0, stddev=1),
              kernel_initializer=TruncatedNormal(mean=0, stddev=1)
        ),
        
        BatchNormalization(),
        Dense(8,  activation = 'elu',
              kernel_regularizer=l1(0.001),
              bias_regularizer=l1(0.001),
              bias_initializer=TruncatedNormal(mean=0, stddev=1),
              kernel_initializer=TruncatedNormal(mean=0, stddev=1.75)
        ),
        Dense(32,
              kernel_regularizer=l2(0.01), 
              bias_regularizer=l1(0.01), 
              bias_initializer=TruncatedNormal(mean=0, stddev=1.5),
              kernel_initializer=TruncatedNormal(mean=0, stddev=4)  
        ),
    
        BatchNormalization(),
        Dense(8, activation = 'elu',
              kernel_regularizer=l1(0.001),
              bias_regularizer=l1(0.001),
              bias_initializer=TruncatedNormal(mean=0, stddev=1),
              kernel_initializer=TruncatedNormal(mean=0, stddev=1.75)
        ),
        
        BatchNormalization(),
        Dense(128, 
              activation = 'elu',
              kernel_regularizer=l1(0.001),
              bias_regularizer=l1(0.001),
              bias_initializer=TruncatedNormal(mean=0, stddev=1.8),
              kernel_initializer=TruncatedNormal(mean=0, stddev=2.5)
        ),
        Dense(1024, 
              kernel_regularizer=l2(0.01), 
              bias_regularizer=l1(0.01), 
              bias_initializer=TruncatedNormal(mean=0, stddev=0.5),
              kernel_initializer=TruncatedNormal(mean=0, stddev=6)
        ),
        Dropout(0.5),
        
        BatchNormalization(),
        Dense(128, 
              activation = 'elu', 
              bias_regularizer=l1(0.001),
              bias_initializer=TruncatedNormal(mean=0, stddev=1.8),
              kernel_initializer=TruncatedNormal(mean=0, stddev=2.5),
              kernel_regularizer=l2(0.001)
        ),
        
        Dense(4, 
              kernel_regularizer=L1L2(0.04, 0.04), 
              bias_regularizer=l2(0.01), 
              bias_initializer=TruncatedNormal(mean=0, stddev=5), 
              kernel_initializer=TruncatedNormal(mean=0, stddev=2.5)
        ),
        Dense(4, 
              kernel_regularizer=L1L2(0.05, 0.05), 
              bias_regularizer=l2(0.1), 
              bias_initializer=TruncatedNormal(mean=0, stddev=0.05), 
              kernel_initializer=TruncatedNormal(mean=0, stddev=4)
        ),
        Dense(4, 
              kernel_regularizer=L1L2(0.6, 0.6), 
              bias_regularizer=l2(0.2), 
              bias_initializer=TruncatedNormal(mean=0, stddev=5), 
              kernel_initializer=TruncatedNormal(mean=0, stddev=2.5)
        ),
        
        Dense(1)
      ])
    
    time_lr = TimeDecayScheduler(learning_rate=0.018, decay_steps=500, decay_rate=0.35, name="")
    
    optimizer = Adam(time_lr)
        
    model.compile(
        loss=MeanSquaredLogarithmicError(name='MSLE'), 
        optimizer=optimizer, 
    )
  
    return model
	