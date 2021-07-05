import numpy as np 
import math
def dpp(kernel_matrix, max_length, epsilon=1E-10):
    """
    Our proposed fast implementation of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    """
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        selected_item = np.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    return selected_items


def extract_ix_dpp(embeds, scores):
    embeds /= np.linalg.norm(embeds, axis=1, keepdims=True)
    similarities = np.dot(embeds, embeds.T)
    kernel_matrix = scores.reshape((embeds.shape[0], 1)) * similarities * scores.reshape((1, embeds.shape[0]))
    result = dpp(kernel_matrix, embeds.shape[0]+1)
    return  result

trials = 10000
probs = [0.1, 0.3, 0.6, 0.87]
 
def temperature_sample(softmax, temperature):
    EPSILON = 10e-16 # to avoid taking the log of zero
    #print(preds)
    softmax = (np.array(softmax) + EPSILON).astype('float64')
    preds = np.log(softmax) / temperature
    #print(preds)
    exp_preds = np.exp(preds)
    #print(exp_preds)
    preds = exp_preds / np.sum(exp_preds)
    #print(preds)
    probas = np.random.multinomial(1, preds, 1)
    assert probas[0].shape[0] == len(softmax)
    return probas[0]#.argmax()

temperatures = [(t or 1) / 100 for t in range(0, 101, 10)]

# for t in temperatures:
#     mean = np.asarray([temperature_sample(probs, t) for _ in range(trials)]).mean(axis=0)
#     print(t, mean)







