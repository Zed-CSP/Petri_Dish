import numpy as np
import tensorflow as tf

# XOR data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

# Create a simple neural network
def create_nn():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation='sigmoid', input_shape=(2,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Evaluate a neural network on XOR
def evaluate_nn(model):
    loss, acc = model.evaluate(X, Y, verbose=0)
    return acc

# Mutate a neural network
def mutate_nn(model):
    for layer in model.layers:
        weights = layer.get_weights()[0]
        biases = layer.get_weights()[1]

        # Apply mutations to weights and biases
        weights += np.random.normal(loc=0, scale=0.5, size=weights.shape)
        biases += np.random.normal(loc=0, scale=0.5, size=biases.shape)

        layer.set_weights([weights, biases])
    return model

# Evolutionary step
def evolve_population(population, n_best, n_random, mutate_chance):
    # Evaluate each individual
    scores = [(evaluate_nn(ind), ind) for ind in population]

    # Sort by score
    scores.sort(key=lambda x: x[0], reverse=True)
    selected = scores[:n_best]

    # Add a few random individuals
    for _ in range(n_random):
        selected.append((0, create_nn()))

    # Breed the next generation
    new_population = []
    while len(new_population) < len(population):
        parent = selected[np.random.randint(len(selected))][1]
        child = tf.keras.models.clone_model(parent)
        child.set_weights(parent.get_weights())

        # Possibly mutate
        if np.random.rand() < mutate_chance:
            child = mutate_nn(child)

        new_population.append(child)

    return new_population

