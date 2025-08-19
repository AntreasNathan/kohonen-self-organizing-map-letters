import numpy as np
import random
from collections import defaultdict

# parameters
gridSizeX = 90  # x-dimension of the grid
gridSizeY = 90  # y-dimension of the grid
learningRate = 0.7  # initial learning rate for weight updates
maxIterations = 100  # maximum number of training epochs
sigma0 = max(gridSizeX, gridSizeY) / 2  # initial neighborhood radius
sigma = sigma0  # neighborhood radius (decays over time)

# load and separate data by letter
data_by_letter = defaultdict(list)
with open("train.txt", "r") as file:
    lines = file.readlines()
    random.shuffle(lines)  # shuffle lines

for line in lines:
    parts = line.split(',')
    letter = parts[0].strip()  # label for each input
    input_values = [int(value) for value in parts[1:17]]
    normalized_inputs = (np.array(input_values) - np.min(input_values)) / (np.max(input_values) - np.min(input_values))
    data_by_letter[letter].append(normalized_inputs)  # normalize values

# split data into train and test sets
train_Data, test_Data, train_Target, test_Target = [], [], [], []
for letter, data in data_by_letter.items():
    split_index = int(len(data) * 0.7)  # 70% training, 30% testing
    train_Data.extend(data[:split_index])
    train_Target.extend([letter] * split_index)
    test_Data.extend(data[split_index:])
    test_Target.extend([letter] * (len(data) - split_index))

train_Data = np.array(train_Data)
test_Data = np.array(test_Data)
weights = np.random.uniform(0, 1, (gridSizeX, gridSizeY, train_Data.shape[1]))  # initialize weights randomly between 0-1

def find_bmu(sample):
    # find the best matching unit (BMU) for a given sample
    distances = np.sum((weights - sample) ** 2, axis=2)
    return np.unravel_index(np.argmin(distances), distances.shape)

def update_weights(sample, bmu_index, epoch):
    # update weights based on the BMU and neighborhood function
    global weights
    learning_rate_decay = learningRate * np.exp(-epoch / maxIterations)  # decay learning rate
    radius_decay = sigma0 * np.exp(-epoch / (maxIterations / np.log(sigma0)))  # decay neighborhood radius
    x, y = np.meshgrid(np.arange(gridSizeX), np.arange(gridSizeY), indexing='ij')
    distance_grid = (x - bmu_index[0]) ** 2 + (y - bmu_index[1]) ** 2  # distance from BMU
    neighborhood_mask = np.exp(-distance_grid / (2 * (radius_decay ** 2)))  # neighborhood influence (Gaussian)
    weights += learning_rate_decay * neighborhood_mask[:, :, np.newaxis] * (sample - weights)  # update weights

def quantization_error(data):
    # calculate quantization error
    distances = [np.min(np.sum((weights - sample) ** 2, axis=2)) for sample in data]
    return np.mean(distances)

def label_network(weights, test_Data, test_labels):
    # label each node in the grid based on test data
    labeled_network = np.full((gridSizeX, gridSizeY), '', dtype=str)
    label_distances = np.full((gridSizeX, gridSizeY), np.inf)  # track minimum distance for each label

    # assign labels to nodes
    for sample, label in zip(test_Data, test_labels):
        distances = np.sum((weights - sample) ** 2, axis=2)  # calculate distances from sample to each node
        for x in range(gridSizeX):
            for y in range(gridSizeY):
                if distances[x, y] < label_distances[x, y]:
                    label_distances[x, y] = distances[x, y]  # update minimum distance
                    labeled_network[x, y] = label  # assign the label of the nearest sample

    return labeled_network

# training loop
errors = []
for epoch in range(maxIterations):
    np.random.shuffle(train_Data)  # shuffle training data each epoch
    for sample in train_Data:
        bmu_index = find_bmu(sample)  # find best matching unit for the sample
        update_weights(sample, bmu_index, epoch)  # update weights based on the BMU
    
    training_error = quantization_error(train_Data)  # training error
    test_error = quantization_error(test_Data)  # test error
    errors.append((epoch + 1, training_error, test_error))  # store errors
    print(f"Epoch: {epoch + 1}, Training Error: {training_error:.4f}, Test Error: {test_error:.4f}")

labeled_network = label_network(weights, test_Data, test_Target)  # label network

# save results
with open("results.txt", "w") as file:
    file.write("Epoch,TrainingError,TestError\n")
    for epoch, training_error, test_error in errors:
        file.write(f"{epoch},{training_error},{test_error}\n")  # save metrics for each epoch in txt

with open("clustering.txt", "w") as file:
    for i in range(labeled_network.shape[0]):
        row = " ".join(labeled_network[i, j] if labeled_network[i, j] else '-' for j in range(labeled_network.shape[1]))
        file.write(row + "\n")  # save labeled network grid to file (clustering.txt)
