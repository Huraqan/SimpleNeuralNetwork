from data.data_handler import load_data_wrapper
from network.SimpleNeuralNetwork import SimpleNeuralNetwork

if not __name__ == "__main__":
    exit()

EPOCHS = 3
SIZES = [28 * 28, 112, 49, 10]

training_data, validation_data, test_data = load_data_wrapper(noise=False)

net : SimpleNeuralNetwork = SimpleNeuralNetwork(sizes=SIZES)
# net : SimpleNeuralNetwork = SimpleNeuralNetwork.load_from_file()

nomenclature = [str(size) for size in SIZES[1:]]
nomenclature = f"{EPOCHS}e_" + "_".join(nomenclature)

net.train_network(training_data, epochs=3, filepath=f"neural_network_{nomenclature}.pickle")
net.validate(test_data, f"costs_{nomenclature}.csv")