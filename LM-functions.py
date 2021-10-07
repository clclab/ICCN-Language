import numpy as np
import torch

def split_to_tokens(encoded_sentence, decoded_sentence):
    """
    Split sentence with separate tokens for punctuation etc.
    (code by Martijn)
    """
    word_ids = encoded_sentence.word_ids()
    token_len = encoded_sentence['attention_mask'].sum().item()
    if transformer in ["gpt2", "gpt2-medium", "gpt2-large"]:
        word_indices = np.array(list(map(lambda e: -1 
            if e is None else e, encoded.word_ids())))[:token_len]
        word_groups = np.split(np.arange(word_indices.shape[0]), 
                               np.unique(word_indices, return_index=True)[1])[1:]
        sent_words = ["".join(list(map(lambda t: t[1:] 
            if t[:1] == "Ġ" else t, np.array(decoded_sentence)[g]))) for g in word_groups]
    else:
        word_indices = np.array(list(map(lambda e: -1 
            if e is None else e, encoded.word_ids())))[1:token_len - 1]
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
