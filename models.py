# Each of the models should return a sries of values that is going to be used to train them
# sicne these values have been previousley determined by testings.
# props : {learning_rate: int, optimizer: Optimizer, }

"""
	Some constant utilites that all the models will be using:
	1. The evaluation metric used for the competition is Mean Squared Logarithmic Error 
		hence all the models are going to be using the MSLE as their evaluation metric.
	2. The Default optimizer will be Adam and RSMprop, also we could choose to use a 
		learning_rate modifier for the Optimizer which would either be a InverseTimeDecay 
		or ExponentialDecay.
	3. They are only few type of layers and activation that work reasonbly for Regression problems:
			Layers: Dense, Dropout, BatchNormalization
			activations: relu, elu, tanh*
	4. For each model, there should be a config dictionary that descrives values that best work for
		the model and this values should have been drived by testing and parameter tuning (potentially
		using keras tuner).
	5. Each function that implements the model gets a full configuration on both compilation and 
		fitting parameters that the model going to need.
	6. The monitored parameter in a EarlyStopping object is always 'val_loss', since the assumption
		is that we will be using validation_data in order to validate our data
	
	Note01: BatchNormalization layers should be used only at the beginning layer of the model since
		it does not work when used in different parts of the model.
	Note02: tanh does not happen to work as well as elu and relu, on a broader perspective relu is 
		most appropiate one for the regression.
"""


# Scheduler objects to control the optimizer learning rate:
from tensorflow.keras.optimizers.schedules import InverseTimeDecay, ExponentialDecay

def TimeDecayScheduler(learning_rate=0.001, decay_steps=200, decay_rate=1.2, name=""):
	""" 
		Returns an InverseTimeDecay object with the given properties to be used in the optimizer.
		
		# params: 
			all the parametes will be needing for a non-staircase InverseTimeDecay Scheduleer
			
		# returns: 
			a InverseTimeDecay object to monitize our optimizer's learning rate
	"""
	return InverseTimeDecay(
		initial_learning_rate=learning_rate, 
		decay_steps=decay_steps,
		decay_rate=decay_rate,
		name=name
    )


def ExonentialScheduler(initial_learning_rate, decay_steps, decay_rate, name=""):
	"""
		Returns an ExponentialDecay object with the given properties to be used in the optimizer.
		
		# params: 
			all the parametes will be needing for a non-staircase ExponentialDecay Scheduleer
			
		# returns: 
			a InverseTimeDecay object to monitize our optimizer's learning rate
	
	"""
	return InverseTimeDecay(
		initial_learning_rate=learning_rate, 
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
			scheduler: If this is passed by the user then use it in the optimizer isntead of the learning rate
		
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
						optimizer isntead of the learning rate
		
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
from tensorflow.keras.regularizers import l2, l1, l1_l2, L1L2  # Regularizers
from tensorflow.keras.losses import MeanSquaredLogarithmicError # Error-metric
import tensorflow_docs as tfdocs # For logging puposes

def Model01(config):
	"""
		# params: 
			config: uses the condifuration dictionary to compile and fit the model accordingly
			
		# returns a history object when the fitting is done
	"""
	pass
	