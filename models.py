def build_model01():
  model = keras.Sequential([
    layers.Dense(64, input_shape=[len(std_d.keys())]),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='softmax'),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='softmax'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='softmax'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.01)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mse', 'msle'])
  return model

def build_model02():
  model = keras.Sequential([
    layers.InputLayer(input_shape=[len(X.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(64),
    layers.Dense(128),
    layers.Dense(256),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    layers.Dense(256),
    layers.Dense(512),
    layers.Dense(1024),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dense(512),
    layers.Dense(256),
    layers.Dense(128),
    layers.BatchNormalization(),
    layers.Dense(64, activation='relu'),
    layers.Dense(64),
    layers.Dense(32),
    layers.Dense(16),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.Adam(0.001)

  model.compile(loss='msle',
                optimizer=optimizer,
                metrics=['msle']
               )
  return model
# 0.16:
def build_model03():
  model = keras.Sequential([
    layers.InputLayer(input_shape=[len(X.keys())]),
    layers.BatchNormalization(),
    layers.Dense(64),
    layers.Dense(64, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.Adam(0.001)

  model.compile(loss='msle',
                optimizer=optimizer,
               )
  return model