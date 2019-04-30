import numpy as np

samples = ['Mortal Kombat 11 released this week.', 'Street Fighter V has been out a while, something like 2016.']

token_index = {}

for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word] = len(token_index) + 1
            
n_samples = len(samples)            
max_length = max([len(sample) for sample in samples])
token_count = max(token_index.values()) + 1
results = np.zeros(shape=(n_samples, max_length, token_count))

for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1.
        
print(results)
