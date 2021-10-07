import numpy as np
import torch
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

transformer = 'distilbert'

def split_to_tokens(encoded_sentence, decoded_sentence):
    """
    Split sentence with separate tokens for punctuation etc.
    (code by Martijn)
    """
    word_ids = encoded_sentence.word_ids()
    token_len = encoded_sentence['attention_mask'].sum().item()
    if transformer in ["gpt2", "gpt2-medium", "gpt2-large"]:
        word_indices = np.array(list(map(lambda e: -1 
            if e is None else e, word_ids)))[:token_len]
        word_groups = np.split(np.arange(word_indices.shape[0]), 
                               np.unique(word_indices, return_index=True)[1])[1:]
        sent_words = ["".join(list(map(lambda t: t[1:] 
            if t[:1] == "Ġ" else t, np.array(decoded_sentence)[g]))) for g in word_groups]
    else:
        word_indices = np.array(list(map(lambda e: -1 
            if e is None else e, word_ids)))[1:token_len - 1]
        word_groups = np.split(np.arange(word_indices.shape[0]) + 1, 
                               np.unique(word_indices, return_index=True)[1])[1:]
        sent_words = ["".join(list(map(lambda t: t[2:] 
            if t[:2] == "##" else t, np.array(decoded_sentence)[g]))) for g in word_groups]
    return word_groups, sent_words

def combine_output_for_layers(model, inputs, states, word_groups, layers):
    """
    Stack embedding layer & hidden states for all layers.
    (code by Martijn)
    """
    if transformer in ["gpt2", "gpt2-medium", "gpt2-large"]:
        emb_layer = model.wte
    else:
        emb_layer = model.embeddings.word_embeddings
        
    sent_tokens_output = torch.stack([
        torch.stack([
                states[i].detach()[:,token_ids_word].mean(axis=1)
                    if i > 0 else
                emb_layer(inputs)[:,token_ids_word].mean(axis=1)
                        for i in layers
            ]).sum(axis=0).squeeze()
                for token_ids_word in word_groups
        ])
    return sent_tokens_output

def my_correlation_rsa(DM1, DM2, method='spearman'):
    """Compute representational similarity between two disimilarity matrices
    """
    # selection elements of the upper triangle
    elements1 = DM1[np.triu_indices(DM1.shape[1],k=1)]
    elements2 = DM2[np.triu_indices(DM2.shape[1],k=1)]

    # compute correlation
    if method == 'pearson':
        correlation_of_similarities = stats.pearsonr(elements1, elements2)
    elif method == 'spearman':
        correlation_of_similarities = stats.spearmanr(elements1, elements2)
    else:
        return NotImplementedError

    return correlation_of_similarities

def RSA_matrix(distance_matrices, method='spearman'):
    # create the matrix to fill with the results
    # I initialized it to ones because the correlation with yourself is always 1
    result_matrix = np.ones((len(distance_matrices), len(distance_matrices)))
    # instead of taking the combinations of pairs of the matrices themselves,
    # take combinations of the indices of the list of matrices (range(3) gives [0,1,2])
    for left_ix, right_ix in combinations(range(len(distance_matrices)), 2):
        # get the matrix that goes with the index
        left = distance_matrices[left_ix]
        right = distance_matrices[right_ix]
        # do the function
        correlation, p_value = my_correlation_rsa(left, right, method=method) 
        # put the result in the matrix
        result_matrix[left_ix][right_ix] = correlation
        # (optionally) also in the other triangle
        result_matrix[right_ix][left_ix] = correlation
    return result_matrix

def plot_RSA(result_matrix, model_name, dist_method):
    layer_labels = ["Embedding","1L","2L","3L","4L","5L","6L", "7L"]
    plt.figure(figsize = (10,6))
    ax = sns.heatmap(result_matrix, annot = True, 
                    cmap = 'magma_r',
                    xticklabels=layer_labels, 
                    yticklabels=layer_labels )
    ax.set_title(f'RSA: {model_name} embeddings across layers for 1 sentence ({dist_method} dist.)')
    plt.show()
    plt.close()
