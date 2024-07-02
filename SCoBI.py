"""A machine learning model that predicts a disease given a list of symptoms."""
# imports
import datasets
import keras
import matplotlib.pyplot as plt  # 3.9.0
import numpy

# Get training dataset.
training_data = datasets.load_dataset("QuyenAnhDE/Diseases_Symptoms", split='train')

# Extract the features you care about from the whole dataset.
symptoms_data = training_data['Symptoms']

illness_data = training_data['Name']

# Preprocess the training data using a TextVectorization layer.
vectorize = keras.layers.TextVectorization()

vectorize.adapt(symptoms_data)

vectorized_symptoms_data = vectorize(symptoms_data)

vectorized_symptoms_data = numpy.array(vectorized_symptoms_data)

vectorize_treatments = keras.layers.TextVectorization()

vectorize_treatments.adapt(illness_data)

vectorize_treatments_data = vectorize_treatments(illness_data)

vectorize_treatments_data = numpy.array(vectorize_treatments_data)

# Make sure the vectorized data is the right type.
print('vectorized_symptoms_data type: ' + str(vectorized_symptoms_data.__class__))

# Confirm the shape of the vectorized data.
print('vectorized_symptoms_data shape: ' + vectorized_symptoms_data.shape.__str__())

for i in range(len(vectorized_symptoms_data)):
    if vectorized_symptoms_data[i] is None:
        numpy.delete(vectorized_symptoms_data, i)
        vectorized_symptoms_data[i] = 0

for i in range(len(vectorized_symptoms_data)):
    if vectorized_symptoms_data[i] is None:
        print('vectorized_symptoms_data[i] value: ' + vectorized_symptoms_data[i].__str__())

# Define a Sequential model.
model = keras.models.Sequential()

# Define model hyperparameters.
batch_size = None
epochs = 10
filters = 10
kernel_size = 2
pool_size = 2

# Define layers for the model.
input_layer = keras.layers.Input(shape=(7,))

# Layer one
dense_layer_one = keras.layers.Dense(units=1140, activation=keras.activations.log_softmax)

# Layer two
reshape_layer_two = keras.layers.Reshape(target_shape=(20, 57))

# Layer Three
layer_three_conv_layer = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')
layer_three_batch_normalization = keras.layers.BatchNormalization()
layer_three_average_pooling = keras.layers.AveragePooling1D(pool_size=pool_size)

# Add layers to the model.
model.add(input_layer)

model.add(dense_layer_one)
model.add(reshape_layer_two)
model.add(layer_three_conv_layer)
model.add(layer_three_batch_normalization)
model.add(layer_three_average_pooling)

# Define the forward pass of the model in a method.

# Define loss function for the model.
loss = keras.losses.CategoricalCrossentropy()

# Define metrics for the model.
metrics = [keras.metrics.CategoricalCrossentropy()]

# Define the optimizer for the model.
optimizer = keras.optimizers.SGD()

# Summarize the model.
model.summary()

# Compile the model.
model.compile(optimizer=optimizer,
              loss=loss,
              metrics=metrics)

# Train the model.
'''model.fit(x=vectorize_treatments_data,
          batch_size=batch_size,
          epochs=epochs)'''

# Plot the loss of the model.
plt.plot(model.losses)

# Plot the metrics of the model.
