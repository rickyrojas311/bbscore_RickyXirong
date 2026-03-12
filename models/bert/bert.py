import os
from collections import deque
from typing import Any, Deque, Dict, List, Union

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class BERT:
    """Loads pre-trained BERT model."""

    def __init__(self, context_length: int = 512):
        """Initializes the Model loader."""
        self.model_mappings = {
            "BERT-Base": "bert-base-uncased",
            "BERT-Large": "bert-large-uncased",
        }
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        # BERT uses [PAD] token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Important: Truncate from the left to keep the target word (usually at the end) in context
        self.tokenizer.truncation_side = 'left'
        
        self.context_length = context_length
        self.static = False
        self._pending_alignment: Deque[Dict[str, Any]] = deque()

    @staticmethod
    def _strip_token(token: str) -> str:
        if token.startswith("##"):
            return token[2:]
        return token

    def _align_tokens_to_words(
        self,
        token_embeddings: torch.Tensor,
        tokens: List[str],
        words: List[str],
    ) -> torch.Tensor:
        stripped_tokens = [self._strip_token(token) for token in tokens]

        aligned_embeddings: List[torch.Tensor] = []
        token_index = 0

        for word in words:
            if word == "":
                continue

            current_pieces: List[str] = []
            current_embeddings: List[torch.Tensor] = []
            
            # Target word matching (uncased for BERT)
            target_word = word.lower()

            while token_index < len(stripped_tokens):
                token = stripped_tokens[token_index]
                
                # Skip special tokens (CLS, SEP, PAD)
                if token in ['[CLS]', '[SEP]', '[PAD]']:
                    token_index += 1
                    continue

                current_pieces.append(token)
                current_embeddings.append(token_embeddings[token_index])
                token_index += 1
                
                if "".join(current_pieces) == target_word:
                    break

            if current_embeddings:
                stacked = torch.stack(current_embeddings, dim=0)
                aligned_embeddings.append(stacked.mean(dim=0))
            else:
                # Fallback: use the closest available embedding
                fallback = token_embeddings[min(
                    token_index, token_embeddings.size(0) - 1)]
                aligned_embeddings.append(fallback)

        if not aligned_embeddings:
            return token_embeddings[-1:].clone()

        return torch.stack(aligned_embeddings, dim=0)

    def preprocess_fn(self, input_data: Union[str, List[str], torch.Tensor]):
        """
        Tokenize input text for BERT.
        """
        if torch.is_tensor(input_data):
            return input_data

        if isinstance(input_data, str):
            input_data = [input_data]

        if isinstance(input_data, list):
            encoded = self.tokenizer(
                input_data,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.context_length,
            )

            all_words = [text.split() for text in input_data]
            for idx, words in enumerate(all_words):
                attention_mask = encoded['attention_mask'][idx]
                valid_length = int(attention_mask.sum().item())
                
                # Store alignment metadata
                self._pending_alignment.append(
                    {
                        'input_ids': encoded['input_ids'][idx],
                        'valid_length': valid_length,
                        'words': words,
                    }
                )

            return encoded['input_ids'].squeeze(0) if len(input_data) == 1 else encoded['input_ids']

        raise ValueError("Input should be a string, list of strings, or tensor.")

    def get_model(self, identifier):
        if identifier not in self.model_mappings:
            raise ValueError(f"Unknown model identifier: {identifier}")
        
        return AutoModel.from_pretrained(self.model_mappings[identifier])

    def postprocess_fn(self, features):
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)

        if features.dim() != 3:
            return features

        aligned_batch: List[torch.Tensor] = []

        for sample_idx in range(features.size(0)):
            if not self._pending_alignment:
                aligned_batch.append(features[sample_idx, -1])
                continue

            metadata = self._pending_alignment.popleft()
            words = metadata['words']
            token_ids = metadata['input_ids']
            
            # Convert IDs to tokens for alignment
            token_list = self.tokenizer.convert_ids_to_tokens(token_ids.tolist())
            
            aligned_embeddings = self._align_tokens_to_words(features[sample_idx], token_list, words)
            aligned_batch.append(aligned_embeddings[-1])

        return torch.stack(aligned_batch, dim=0)
