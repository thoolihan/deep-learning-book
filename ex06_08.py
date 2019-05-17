import numpy as np
from shared.logger import get_logger

logger = get_logger()

timesteps = 100
input_features = 32
output_features = 64

inputs = np.random.random((timesteps, input_features))
logger.info("Created inputs with shape: {}".format(inputs.shape))

state_t = np.zeros((output_features,))
logger.info("Created state_t with shape: {}".format(state_t.shape))

W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features,))

successive_outputs = []
for input_t in inputs:
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
    successive_outputs.append(output_t)
    state_t = output_t
    timestep = len(successive_outputs) 
    if timestep % 5 == 0:
        logger.info("Processed timestep: {}".format(timestep))

final_output_sequence = np.concatenate(successive_outputs, axis=0)