
 ######################################################

# Copyright (c) OpenMMLab. All rights reserved.
# from collections import OrderedDict
# from typing import Sequence
# from types import SimpleNamespace

# import torch
# from mmengine.model import BaseModel
# from torch import nn

# try:
#     import open_clip
# except ImportError:
#     open_clip = None

# from mmdet.registry import MODELS


# @MODELS.register_module()
# class CLIPTextModel(BaseModel):
#     """CLIP model for language embedding using OpenCLIP.

#     Args:
#         name (str): CLIP model name.
#         pretrained (str): Pretrained weights name, e.g. 'openai'.
#         max_tokens (int): Max token length (CLIP uses 77).
#         use_sub_sentence_represent (bool): Enable sub-sentence representation.
#         special_tokens_list (list): List of special tokens for sentence splits.
#         num_layers_of_embedded (int): How many hidden layers to average.
#         pad_to_max (bool): Whether to pad input to max length.
#         add_pooling_layer (bool): Whether to use pooling (not used here).
#     """

#     def __init__(self,
#                  name: str = 'ViT-B-32-quickgelu',
#                  pretrained: str = 'openai',
#                  max_tokens: int = 77,
#                  use_sub_sentence_represent: bool = False,
#                  special_tokens_list: list = None,
#                  num_layers_of_embedded: int = 1,
#                  pad_to_max: bool = False,
#                  add_pooling_layer: bool = False,
#                  **kwargs):

#         # Remove extra arguments from kwargs before passing to BaseModel
#         kwargs.pop('pad_to_max', None)
#         kwargs.pop('use_sub_sentence_represent', None)
#         kwargs.pop('special_tokens_list', None)
#         kwargs.pop('add_pooling_layer', None)

#         super().__init__(**kwargs)

#         if open_clip is None:
#             raise ImportError('open_clip not found. Install it with: pip install open_clip_torch')

#         self.max_tokens = max_tokens
#         self.use_sub_sentence_represent = use_sub_sentence_represent
#         self.pad_to_max = pad_to_max
#         self.add_pooling_layer = add_pooling_layer
#         self.num_layers_of_embedded = num_layers_of_embedded

#         # Load CLIP model and tokenizer
#         self.model, _, _ = open_clip.create_model_and_transforms(
#             model_name=name, pretrained=pretrained)
#         self.tokenizer = open_clip.get_tokenizer(name)

#         self.set_requires_grad(False)

#         self.language_dim = self.model.text_projection.shape[1]

#         # For compatibility with MMDet
#         self.language_backbone = SimpleNamespace(
#             body=SimpleNamespace(language_dim=self.language_dim)
#         )

#         if self.use_sub_sentence_represent:
#             assert special_tokens_list is not None, \
#                 'special_tokens_list must be set if use_sub_sentence_represent is True'
#             self.special_tokens = self.tokenizer(
#                 special_tokens_list, context_length=self.model.context_length)

#     def forward(self, captions: Sequence[str], **kwargs) -> dict:
#         """Forward pass to compute CLIP text embeddings."""
#         device = next(self.model.parameters()).device

#         tokenized = self.tokenizer(
#             captions,
#             context_length=self.model.context_length
#         ).to(device)

#         outputs = self.model.encode_text(tokenized)  # [B, D]

#         results = {
#             'embedded': outputs,                # [B, D]
#             'masks': torch.ones_like(outputs),  # Dummy mask
#             'hidden': outputs.unsqueeze(1)      # [B, 1, D]
#         }

#         if self.use_sub_sentence_represent:
#             results['text_token_mask'] = torch.ones_like(tokenized, dtype=torch.bool)

#         return results

#     def set_requires_grad(self, requires_grad: bool = True, freeze_projection: bool = False):
#         """Enable or disable gradients for CLIP text encoder.

#         Args:
#             requires_grad (bool): Enable gradients for the model.
#             freeze_projection (bool): Keep the text_projection layer frozen.
#         """
#         for name, param in self.model.named_parameters():
#             if freeze_projection and 'text_projection' in name:
#                 param.requires_grad = False
#             else:
#                 param.requires_grad = requires_grad

#####################################################
from collections import OrderedDict
from typing import Sequence
from types import SimpleNamespace

import torch
from mmengine.model import BaseModel
from torch import nn

from transformers import CLIPTokenizer, CLIPModel

from mmdet.registry import MODELS


@MODELS.register_module()
class CLIPTextModel(BaseModel):
    """CLIP model for language embedding using HuggingFace CLIP.

    Args:
        name (str): Model name.
        pretrained (str): Pretrained checkpoint name or path.
        max_tokens (int): Max token length.
        use_sub_sentence_represent (bool): Enable sub-sentence representation.
        special_tokens_list (list): Special tokens for sub-sentence split.
        num_layers_of_embedded (int): Number of hidden layers to average.
        pad_to_max (bool): Pad input to max length.
        add_pooling_layer (bool): Not used.
    """

    def __init__(self,
                 name: str = 'openai/clip-vit-base-patch32',
                 pretrained: str = 'openai/clip-vit-base-patch32',
                 max_tokens: int = 77,
                 use_sub_sentence_represent: bool = False,
                 special_tokens_list: list = None,
                 num_layers_of_embedded: int = 1,
                 pad_to_max: bool = False,
                 add_pooling_layer: bool = False,
                 **kwargs):

        # Remove extra arguments before passing to BaseModel
        kwargs.pop('pad_to_max', None)
        kwargs.pop('use_sub_sentence_represent', None)
        kwargs.pop('special_tokens_list', None)
        kwargs.pop('add_pooling_layer', None)

        super().__init__(**kwargs)

        self.max_tokens = max_tokens
        self.use_sub_sentence_represent = use_sub_sentence_represent
        self.pad_to_max = pad_to_max
        self.add_pooling_layer = add_pooling_layer
        self.num_layers_of_embedded = num_layers_of_embedded

        # Load HuggingFace CLIP
        self.tokenizer = CLIPTokenizer.from_pretrained(name)
        self.model = CLIPModel.from_pretrained(pretrained)

        self.set_requires_grad(False)

        self.language_dim = self.model.text_model.config.hidden_size

        # MMDet compatibility
        self.language_backbone = SimpleNamespace(
            body=SimpleNamespace(language_dim=self.language_dim)
        )

        if self.use_sub_sentence_represent:
            assert special_tokens_list is not None, \
                'special_tokens_list must be set if use_sub_sentence_represent is True'
            self.special_tokens = self.tokenizer(
                special_tokens_list, padding='max_length',
                max_length=self.max_tokens,
                truncation=True,
                return_tensors="pt"
            )

    def forward(self, captions: Sequence[str], **kwargs) -> dict:
        """Forward pass to compute text embeddings."""
        device = next(self.model.parameters()).device

        tokenized = self.tokenizer(
            captions,
            padding='max_length' if self.pad_to_max else True,
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_token_type_ids=True,
            return_attention_mask=True
        ).to(device)

        outputs = self.model.get_text_features(**tokenized)  # [B, D]

        results = {
            'embedded': outputs,                # [B, D]
            'masks': torch.ones_like(outputs),  # dummy masks
            'hidden': outputs.unsqueeze(1),     # [B, 1, D]
            'tokenized': tokenized              # include tokenizer output for char_to_token()
        }

        if self.use_sub_sentence_represent:
            results['text_token_mask'] = tokenized.attention_mask.bool()

        return results

    def set_requires_grad(self, requires_grad: bool = True, freeze_projection: bool = False):
        """Enable or disable gradients for the CLIP text encoder."""
        for name, param in self.model.named_parameters():
            if freeze_projection and 'text_projection' in name:
                param.requires_grad = False
            else:
                param.requires_grad = requires_grad


###############################################

# from typing import Sequence, Optional, Dict
# from types import SimpleNamespace

# import torch
# from torch import nn
# from mmengine.model import BaseModel

# from transformers import CLIPTokenizer, CLIPTextModel as HFCLIPTextModel

# from mmdet.registry import MODELS


# @MODELS.register_module()
# class CLIPTextModel(BaseModel):
#     def __init__(self,
#                  name: str = 'ViT-B-32-quickgelu',
#                  pretrained_model_name_or_path: str = 'openai/clip-vit-base-patch32',
#                  max_tokens: int = 77,
#                  use_sub_sentence_represent: bool = False,
#                  special_tokens_list: Optional[list] = None,
#                  num_layers_of_embedded: int = 1,
#                  pad_to_max: bool = False,
#                  add_pooling_layer: bool = False,
#                  **kwargs):
#         kwargs.pop('pad_to_max', None)
#         kwargs.pop('use_sub_sentence_represent', None)
#         kwargs.pop('special_tokens_list', None)
#         kwargs.pop('add_pooling_layer', None)

#         super().__init__(**kwargs)

#         self.max_tokens = max_tokens
#         self.use_sub_sentence_represent = use_sub_sentence_represent
#         self.pad_to_max = pad_to_max
#         self.add_pooling_layer = add_pooling_layer
#         self.num_layers_of_embedded = num_layers_of_embedded

#         self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=True)
#         self.model = HFCLIPTextModel.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)

#         self.set_requires_grad(False)

#         self.language_dim = self.model.config.hidden_size
#         self.language_backbone = SimpleNamespace(
#             body=SimpleNamespace(language_dim=self.language_dim)
#         )

#     def tokenize(self, captions: Sequence[str]) -> Dict[str, torch.Tensor]:
#         """Tokenize text outside the forward pass for GroundingDINO compatibility."""
#         return self.tokenizer(
#             captions,
#             padding='max_length' if self.pad_to_max else True,
#             truncation=True,
#             max_length=self.max_tokens,
#             return_tensors='pt'
#         )

#     def forward(self, tokenized_text: Dict[str, torch.Tensor], **kwargs) -> dict:
#         """
#         Forward pass expects tokenized input dict with keys 'input_ids' and 'attention_mask'.

#         Args:
#             tokenized_text (dict): Tokenized text inputs from tokenizer.

#         Returns:
#             dict: Dictionary with keys 'embedded', 'masks', and 'hidden'.
#         """
#         device = next(self.model.parameters()).device

#         input_ids = tokenized_text['input_ids'].to(device)
#         attention_mask = tokenized_text['attention_mask'].to(device)

#         outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

#         if outputs.pooler_output is not None:
#             embedded = outputs.pooler_output
#         else:
#             embedded = outputs.last_hidden_state.mean(dim=1)

#         results = {
#             'embedded': embedded,
#             'masks': attention_mask,
#             'hidden': outputs.last_hidden_state,
#         }

#         if self.use_sub_sentence_represent:
#             results['text_token_mask'] = attention_mask.bool()

#         return results

#     def set_requires_grad(self, requires_grad: bool = True, freeze_projection: bool = False):
#         """Freeze or unfreeze all parameters."""
#         for name, param in self.model.named_parameters():
#             param.requires_grad = requires_grad
