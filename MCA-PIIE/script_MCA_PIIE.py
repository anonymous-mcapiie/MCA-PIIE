"""
MCA-PIIE: Multi-Context Attention for PII Extraction
=====================================================
Faithful reimplementation based on the manuscript:
"A Decision Support Framework for Privacy Risk Assessment on Social Media
 through Automated Personally Identifiable Information Detection"

Architecture (per Figure 4 in the paper):
  Input Representation:
    - Character Bi-LSTM  -> e_c
    - Word Embeddings (GloVe-Twitter-200d) -> e_w
    - Dependency Graph (syntax-level)

  PII Attention Mechanisms:
    - Self Attention (Transformer): operates on E = [e_c; e_w] -> S
    - PII-GAT (Graph Attention Network): operates on e_w + dependency graph -> G
    - Highway Layer: fuses S and G -> u
    - Gating Mechanism: combines E and u -> v
    - Global Attention: re-weights v by PII-indicative salience -> R

  Classification:
    - CRF layer on R -> BIOES labels

  Deep Transfer Learning:
    - Phase 1: Train on source domain (6 public datasets)
    - Phase 2: Transfer Character Bi-LSTM, PII-GAT, Transformer, Global Attention
               to target model; fine-tune input/output layers on Twitter data
"""

import os
import random
import json
import math
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from collections import Counter, defaultdict
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Try imports for optional dependencies
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except (ImportError, OSError):
    SPACY_AVAILABLE = False
    print("Warning: spaCy not available. Dependency parsing will use fallback.")

try:
    from nltk.tokenize import word_tokenize
    import nltk
    nltk.data.find('tokenizers/punkt_tab')
    NLTK_AVAILABLE = True
except (ImportError, LookupError):
    NLTK_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

class Config:
    # PII categories (7 types per paper)
    PII_TYPES = ["age", "contact", "date", "ID", "location", "name", "profession"]

    # BIOES tagging
    TAGGING_SCHEME = "BIOES"  # B=Beginning, I=Inside, O=Outside, E=Ending, S=Single

    # Character embeddings
    CHAR_EMBED_DIM = 30
    CHAR_HIDDEN_DIM = 50  # Bi-LSTM output will be 2 * CHAR_HIDDEN_DIM = 100

    # Word embeddings (GloVe-Twitter-200d)
    WORD_EMBED_DIM = 200
    GLOVE_FILE = "glove.twitter.27B.200d.txt"  # path to GloVe file

    # Model dimensions
    # E = [e_c; e_w], so input dim = 2*CHAR_HIDDEN_DIM + WORD_EMBED_DIM = 300
    INPUT_DIM = 2 * CHAR_HIDDEN_DIM + WORD_EMBED_DIM  # 300
    HIDDEN_DIM = 300  # Keep consistent through the model

    # Transformer (Self Attention)
    TRANSFORMER_HEADS = 6
    TRANSFORMER_LAYERS = 2
    TRANSFORMER_FF_DIM = 512
    TRANSFORMER_DROPOUT = 0.1

    # PII-GAT
    GAT_HEADS = 4
    GAT_HIDDEN_DIM = 64  # per head, so output = GAT_HEADS * GAT_HIDDEN_DIM before projection
    # a=2: explicit 2-hop adjacency computed at runtime (no layer stacking needed)
    GAT_DROPOUT = 0.1

    # Global Attention
    GLOBAL_ATT_DIM = 300

    # Training
    MAX_SEQ_LENGTH = 128
    MAX_WORD_LENGTH = 30
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 30
    SOURCE_EPOCHS = 30          # Same as NUM_EPOCHS by default
    PATIENCE = 5                # early stopping
    DROPOUT = 0.3
    GRAD_CLIP = 5.0
    K_FOLDS = 5                 # 5-fold cross-validation (per paper Section 5.3)
    MAX_SOURCE_SAMPLES = None   # None = use all source data (paper default)
                                # Set to e.g. 50000 for faster debugging

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_label_mappings(pii_types=None):
    """Create BIOES label mappings."""
    if pii_types is None:
        pii_types = Config.PII_TYPES

    labels = ["O"]
    for pii_type in pii_types:
        for prefix in ["B", "I", "E", "S"]:
            labels.append(f"{prefix}-{pii_type}")

    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label, labels


LABEL2ID, ID2LABEL, ALL_LABELS = create_label_mappings()
NUM_LABELS = len(ALL_LABELS)


# =============================================================================
# CRF Layer (per Section 5.2.3)
# =============================================================================

class CRF(nn.Module):
    """
    Conditional Random Field layer for sequence labeling.

    Per the paper (Section 5.2.3):
      score(T, y) = sum_i R_{i,y_i} + sum_i A_{y_i, y_{i+1}}
      P(y|T) = exp(score(T,y)) / sum_{y'} exp(score(T,y'))

    where A is the transition matrix.
    """

    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags

        # Transition matrix A: A[i,j] = score of transitioning from tag i to tag j
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))

        # Start and end transition scores
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

        self._init_constraints()

    def _init_constraints(self):
        """Initialize with BIOES constraints: prevent invalid transitions."""
        with torch.no_grad():
            # Set invalid transitions to large negative value
            NEG_INF = -10000.0

            for i, tag_from in ID2LABEL.items():
                for j, tag_to in ID2LABEL.items():
                    # I-X or E-X can only follow B-X or I-X (same X)
                    if tag_to.startswith("I-") or tag_to.startswith("E-"):
                        type_to = tag_to[2:]
                        if tag_from == "O" or tag_from.startswith("S-") or tag_from.startswith("E-"):
                            self.transitions.data[i, j] = NEG_INF
                        elif (tag_from.startswith("B-") or tag_from.startswith("I-")):
                            type_from = tag_from[2:]
                            if type_from != type_to:
                                self.transitions.data[i, j] = NEG_INF

                    # After B-X or I-X, must be I-X or E-X (same X)
                    if tag_from.startswith("B-") or tag_from.startswith("I-"):
                        type_from = tag_from[2:]
                        if tag_to == "O" or tag_to.startswith("B-") or tag_to.startswith("S-"):
                            self.transitions.data[i, j] = NEG_INF
                        elif (tag_to.startswith("I-") or tag_to.startswith("E-")):
                            type_to = tag_to[2:]
                            if type_from != type_to:
                                self.transitions.data[i, j] = NEG_INF

            # Start constraints: cannot start with I-X or E-X
            for i, tag in ID2LABEL.items():
                if tag.startswith("I-") or tag.startswith("E-"):
                    self.start_transitions.data[i] = NEG_INF

            # End constraints: cannot end with B-X or I-X
            for i, tag in ID2LABEL.items():
                if tag.startswith("B-") or tag.startswith("I-"):
                    self.end_transitions.data[i] = NEG_INF

    def forward(self, emissions, tags, mask):
        """Compute negative log-likelihood loss."""
        # emissions: [batch, seq_len, num_tags]
        # tags: [batch, seq_len]
        # mask: [batch, seq_len] (1 = valid, 0 = padding)

        log_likelihood = self._compute_log_likelihood(emissions, tags, mask)
        return -log_likelihood.mean()

    def _compute_log_likelihood(self, emissions, tags, mask):
        batch_size, seq_len, _ = emissions.shape

        # Numerator: score of the correct path
        score = self._compute_score(emissions, tags, mask)

        # Denominator: log-sum-exp of all paths (forward algorithm)
        log_partition = self._compute_log_partition(emissions, mask)

        return score - log_partition

    def _compute_score(self, emissions, tags, mask):
        batch_size, seq_len, _ = emissions.shape

        score = self.start_transitions[tags[:, 0]]
        score += emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)

        for i in range(1, seq_len):
            current_mask = mask[:, i]
            transition_score = self.transitions[tags[:, i - 1], tags[:, i]]
            emission_score = emissions[:, i].gather(1, tags[:, i].unsqueeze(1)).squeeze(1)
            score += (transition_score + emission_score) * current_mask

        # End transition
        last_tag_indices = mask.long().sum(dim=1) - 1
        last_tags = tags.gather(1, last_tag_indices.unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_tags]

        return score

    def _compute_log_partition(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.shape

        # Initialize with start transitions + first emissions
        alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]  # [batch, num_tags]

        for i in range(1, seq_len):
            current_mask = mask[:, i].unsqueeze(1)  # [batch, 1]

            # alpha_expand: [batch, num_tags, 1]
            # trans: [num_tags, num_tags]
            # emit: [batch, 1, num_tags]
            alpha_expand = alpha.unsqueeze(2)  # [batch, num_tags, 1]
            trans = self.transitions.unsqueeze(0)  # [1, num_tags, num_tags]
            emit = emissions[:, i].unsqueeze(1)  # [batch, 1, num_tags]

            scores = alpha_expand + trans + emit  # [batch, num_tags, num_tags]
            new_alpha = torch.logsumexp(scores, dim=1)  # [batch, num_tags]

            alpha = new_alpha * current_mask + alpha * (1 - current_mask)

        # Add end transitions
        alpha = alpha + self.end_transitions.unsqueeze(0)

        return torch.logsumexp(alpha, dim=1)

    def decode(self, emissions, mask):
        """Viterbi decoding to find the best tag sequence."""
        batch_size, seq_len, num_tags = emissions.shape

        # Initialize
        viterbi_score = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        viterbi_path = []

        for i in range(1, seq_len):
            current_mask = mask[:, i].unsqueeze(1)

            score_expand = viterbi_score.unsqueeze(2)  # [batch, num_tags, 1]
            trans = self.transitions.unsqueeze(0)  # [1, num_tags, num_tags]

            scores = score_expand + trans  # [batch, num_tags, num_tags]
            best_scores, best_paths = scores.max(dim=1)  # [batch, num_tags]

            emit = emissions[:, i]
            new_score = best_scores + emit

            viterbi_score = new_score * current_mask + viterbi_score * (1 - current_mask)
            viterbi_path.append(best_paths)

        # End transitions
        viterbi_score = viterbi_score + self.end_transitions.unsqueeze(0)

        # Backtrack
        best_last_tags = viterbi_score.argmax(dim=1)  # [batch]

        best_paths_list = []
        for b in range(batch_size):
            path = [best_last_tags[b].item()]
            seq_length = int(mask[b].sum().item())

            for i in range(len(viterbi_path) - 1, -1, -1):
                if i + 1 < seq_length:
                    path.append(viterbi_path[i][b, path[-1]].item())

            path.reverse()
            # Pad to seq_len
            path = path[:seq_length] + [0] * (seq_len - seq_length)
            best_paths_list.append(path)

        return best_paths_list


# =============================================================================
# Character Bi-LSTM (per Section 5.2.1)
# =============================================================================

class CharBiLSTM(nn.Module):
    """
    Character-level Bi-LSTM to capture morphological features.

    Per Section 5.2.1:
      "Each word's characters are embedded and processed through a Bi-LSTM,
       producing character-based representations (e_c)."

    Output dim: 2 * char_hidden_dim
    """

    def __init__(self, char_vocab_size, char_embed_dim, char_hidden_dim, dropout=0.1):
        super().__init__()
        self.char_embed = nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=0)
        self.char_lstm = nn.LSTM(
            char_embed_dim, char_hidden_dim,
            batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, char_ids, char_lengths):
        """
        Args:
            char_ids: [batch, seq_len, max_word_len]
            char_lengths: [batch, seq_len] actual character lengths per word
        Returns:
            char_repr: [batch, seq_len, 2*char_hidden_dim]
        """
        batch_size, seq_len, max_word_len = char_ids.shape

        # Reshape for processing all words at once
        char_ids_flat = char_ids.view(-1, max_word_len)  # [batch*seq_len, max_word_len]
        char_lengths_flat = char_lengths.view(-1)  # [batch*seq_len]

        # Embed characters
        char_embeds = self.dropout(self.char_embed(char_ids_flat))  # [batch*seq_len, max_word_len, char_embed_dim]

        # Clamp lengths to avoid zero-length sequences
        char_lengths_clamped = char_lengths_flat.clamp(min=1)

        # Pack, run LSTM, unpack
        packed = pack_padded_sequence(
            char_embeds, char_lengths_clamped.cpu(),
            batch_first=True, enforce_sorted=False
        )
        lstm_out, (hidden, _) = self.char_lstm(packed)

        # Use final hidden states from both directions
        # hidden: [2, batch*seq_len, char_hidden_dim]
        char_repr = torch.cat([hidden[0], hidden[1]], dim=-1)  # [batch*seq_len, 2*char_hidden_dim]

        char_repr = char_repr.view(batch_size, seq_len, -1)  # [batch, seq_len, 2*char_hidden_dim]

        return char_repr


# =============================================================================
# PII-GAT: Graph Attention Network (per Section 5.2.2)
# =============================================================================

class GATLayer(nn.Module):
    """
    Single Graph Attention layer.

    Per paper:
      g_i = MultiAtt(t_i, {forall a[t_i | edge_{k,i}]})

    With multi-head attention over dependency neighbors.
    """

    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1, concat=True):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.concat = concat

        self.W = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.a_src = nn.Parameter(torch.FloatTensor(num_heads, out_dim))
        self.a_dst = nn.Parameter(torch.FloatTensor(num_heads, out_dim))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x, adj):
        """
        Args:
            x: [seq_len, in_dim]
            adj: [seq_len, seq_len] adjacency matrix
        Returns:
            out: [seq_len, out_dim * num_heads] if concat, else [seq_len, out_dim]
        """
        N = x.size(0)

        # Linear transformation
        h = self.W(x)  # [N, out_dim * num_heads]
        h = h.view(N, self.num_heads, self.out_dim)  # [N, heads, out_dim]

        # Attention scores
        # a_src, a_dst: [heads, out_dim]
        attn_src = (h * self.a_src.unsqueeze(0)).sum(dim=-1)  # [N, heads]
        attn_dst = (h * self.a_dst.unsqueeze(0)).sum(dim=-1)  # [N, heads]

        # Pairwise attention: e_ij = LeakyReLU(a_src_i + a_dst_j)
        attn = attn_src.unsqueeze(1) + attn_dst.unsqueeze(0)  # [N, N, heads]
        attn = self.leaky_relu(attn)

        # Mask non-neighbors using adjacency matrix
        # adj: [N, N], expand to [N, N, heads]
        mask = adj.unsqueeze(-1).expand_as(attn)
        attn = attn.masked_fill(mask == 0, float('-inf'))

        # Softmax over neighbors
        attn = F.softmax(attn, dim=1)
        attn = torch.nan_to_num(attn, nan=0.0)  # handle all-masked rows
        attn = self.dropout(attn)

        # Aggregate: weighted sum of neighbor features
        # attn: [N, N, heads], h: [N, heads, out_dim]
        out = torch.einsum('ijh,jhd->ihd', attn, h)  # [N, heads, out_dim]

        if self.concat:
            out = out.reshape(N, self.num_heads * self.out_dim)
        else:
            out = out.mean(dim=1)  # [N, out_dim]

        return out


class PIIGAT(nn.Module):
    """
    PII-GAT: Graph Attention Network for syntactic patterns.

    Per paper (Section 5.2.2):
      "a is set to 2 to include all 2-step neighboring words to obtain
       a wider scope of syntactic patterns"

    Implementation:
      We explicitly compute the 2-hop adjacency matrix A_2hop = (A + A^2 > 0)
      so that each node attends to both its direct neighbors AND neighbors-
      of-neighbors. A single GAT layer then performs multi-head attention
      over this expanded neighborhood. This is more faithful to the paper's
      description than implicitly stacking multiple GAT layers.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4, dropout=0.1):
        super().__init__()

        # Single GAT layer over the 2-hop adjacency
        self.gat_layer = GATLayer(
            input_dim, hidden_dim, num_heads, dropout, concat=True
        )

        # Project concatenated multi-head output to desired output_dim
        self.proj = nn.Linear(hidden_dim * num_heads, output_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def compute_2hop_adj(adj):
        """
        Compute explicit 2-hop adjacency: A_2hop = (A + A^2 > 0).
        Nodes can attend to direct neighbors AND 2-step neighbors.
        """
        adj_sq = torch.matmul(adj, adj)  # A^2: 2-hop connections
        adj_2hop = ((adj + adj_sq) > 0).float()  # Union of 1-hop and 2-hop
        return adj_2hop

    def forward(self, x, adj):
        """
        Args:
            x: [seq_len, input_dim] word embeddings
            adj: [seq_len, seq_len] dependency adjacency matrix
        Returns:
            G: [seq_len, output_dim] syntactic representations
        """
        # Compute explicit 2-hop adjacency
        if not isinstance(adj, torch.Tensor):
            adj = torch.tensor(adj, dtype=torch.float, device=x.device)
        adj_2hop = self.compute_2hop_adj(adj)

        # Single GAT layer over expanded neighborhood
        h = self.gat_layer(x, adj_2hop)  # [seq_len, hidden_dim * num_heads]
        h = F.elu(h)
        h = self.dropout(h)

        # Project to output dim
        G = self.proj(h)  # [seq_len, output_dim]

        return G


# =============================================================================
# Highway Layer (per Section 5.2.2)
# =============================================================================

class HighwayLayer(nn.Module):
    """
    Highway Layer for fusing Self Attention (S) and Graph Attention (G) outputs.

    Per paper:
      T = sigma(W_T^T x + b_T)
      Highway(x) = T * FFN(x, W_H) + (1-T) * x
      u = Highway([S, G, |S-G|, S⊙G])
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Input is [S, G, |S-G|, S⊙G], so 4 * dim
        self.ffn = nn.Linear(input_dim * 4, output_dim)
        self.gate = nn.Linear(input_dim * 4, output_dim)

    def forward(self, S, G):
        """
        Args:
            S: [batch, seq_len, dim] - Self Attention output
            G: [batch, seq_len, dim] - GAT output
        Returns:
            u: [batch, seq_len, output_dim]
        """
        # Concatenate: [S, G, |S-G|, S⊙G]
        combined = torch.cat([S, G, torch.abs(S - G), S * G], dim=-1)

        T = torch.sigmoid(self.gate(combined))
        H = F.relu(self.ffn(combined))

        u = T * H + (1 - T) * S  # Use S as the residual (semantic representation)

        return u


# =============================================================================
# Gating Mechanism (per Section 5.2.2)
# =============================================================================

class GatingMechanism(nn.Module):
    """
    Gating mechanism to combine original embeddings E with contextual
    representations u.

    Per paper:
      g = sigma(W1 * E + W2 * u + b)
      v = g * E + (1-g) * u

    "As g increases, the output representation v relies more on the original
     embeddings E but less on the contextual representations u."
    """

    def __init__(self, dim):
        super().__init__()
        self.W1 = nn.Linear(dim, dim)
        self.W2 = nn.Linear(dim, dim)

    def forward(self, E, u):
        """
        Args:
            E: [batch, seq_len, dim] - original input embeddings
            u: [batch, seq_len, dim] - fused contextual representations
        Returns:
            v: [batch, seq_len, dim]
        """
        g = torch.sigmoid(self.W1(E) + self.W2(u))
        v = g * E + (1 - g) * u
        return v


# =============================================================================
# Global Attention (per Section 5.2.2)
# =============================================================================

class GlobalAttention(nn.Module):
    """
    Global Attention mechanism for emphasizing PII-indicative terms.

    Per paper (Section 5.2.2):
      lambda = softmax((v)^T * c_x)
      R = v * lambda^T

    "Global Attention examines the relationships between each word in the
     input text and the output, assigning higher weights to terms highly
     correlated with PIIE, such as 'name', 'location', and 'phone number'."

    Implementation:
      c_x is a learnable context vector (per the paper, it determines
      the salience of each token for PII extraction).
      Softmax is computed over the **sequence dimension** (dim=1), so that
      lambda represents how important each token is relative to all other
      tokens in the sentence — this is the standard "global attention"
      formulation where high-salience PII-indicative terms (e.g., "phone",
      "name", "address") receive higher weights.

      R = v * lambda^T re-scales each token's representation by its
      global salience weight.
    """

    def __init__(self, dim):
        super().__init__()
        # c_x: learnable context vector [dim]
        # Computes salience score for each token: score_i = v_i^T * c_x
        self.c_x = nn.Parameter(torch.randn(dim))
        nn.init.xavier_uniform_(self.c_x.unsqueeze(0))

    def forward(self, v, mask=None):
        """
        Args:
            v: [batch, seq_len, dim] - gated representation
            mask: [batch, seq_len] - 1=valid, 0=padding (optional)
        Returns:
            R: [batch, seq_len, dim] - globally attended representation
        """
        # Compute salience scores: each token's alignment with the context vector
        # scores_i = v_i^T * c_x -> scalar per token
        scores = torch.matmul(v, self.c_x)  # [batch, seq_len]

        # Mask padding positions before softmax
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax over sequence dimension: which tokens are most salient?
        lam = F.softmax(scores, dim=1)  # [batch, seq_len]

        # R = v * lambda^T (re-scale each token by its salience weight)
        lam = lam.unsqueeze(-1)  # [batch, seq_len, 1]
        R = v * lam  # [batch, seq_len, dim]

        return R


# =============================================================================
# Full MCA-PIIE Model
# =============================================================================

class MCAPIIE(nn.Module):
    """
    Multi-Context Attention for PII Extraction (MCA-PIIE).

    Architecture per Figure 4:
      1. Input Representation:
         - Character Bi-LSTM -> e_c [batch, seq_len, 2*char_hidden]
         - Word Embeddings (GloVe) -> e_w [batch, seq_len, word_embed_dim]
         - E = [e_c; e_w]  (concatenation)
         - Dependency graph for syntax

      2. PII Attention Mechanisms:
         - Self Attention (Transformer): E -> S
         - PII-GAT: e_w + adj -> G
         - Highway Layer: (S, G) -> u
         - Gating: (E, u) -> v
         - Global Attention: v -> R

      3. CRF: R -> BIOES labels
    """

    def __init__(self, config, char_vocab_size, word_vocab_size,
                 pretrained_word_embeddings=None):
        super().__init__()
        self.config = config

        # --- Input Representation ---

        # Character Bi-LSTM
        self.char_bilstm = CharBiLSTM(
            char_vocab_size=char_vocab_size,
            char_embed_dim=config.CHAR_EMBED_DIM,
            char_hidden_dim=config.CHAR_HIDDEN_DIM,
            dropout=config.DROPOUT
        )

        # Word embeddings (GloVe-Twitter-200d)
        self.word_embed = nn.Embedding(word_vocab_size, config.WORD_EMBED_DIM, padding_idx=0)
        if pretrained_word_embeddings is not None:
            self.word_embed.weight.data.copy_(pretrained_word_embeddings)
            # Allow fine-tuning during target domain training

        self.input_dropout = nn.Dropout(config.DROPOUT)

        # --- Self Attention (Transformer) ---
        # Operates on E = [e_c; e_w]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.INPUT_DIM,
            nhead=config.TRANSFORMER_HEADS,
            dim_feedforward=config.TRANSFORMER_FF_DIM,
            dropout=config.TRANSFORMER_DROPOUT,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.TRANSFORMER_LAYERS
        )

        # --- PII-GAT ---
        # Operates on word embeddings + dependency graph
        self.pii_gat = PIIGAT(
            input_dim=config.WORD_EMBED_DIM,
            hidden_dim=config.GAT_HIDDEN_DIM,
            output_dim=config.INPUT_DIM,  # Match dimension for Highway fusion
            num_heads=config.GAT_HEADS,
            dropout=config.GAT_DROPOUT
        )

        # --- Highway Layer ---
        # Fuses Self Attention output S and GAT output G
        self.highway = HighwayLayer(
            input_dim=config.INPUT_DIM,
            output_dim=config.INPUT_DIM
        )

        # --- Gating Mechanism ---
        self.gating = GatingMechanism(config.INPUT_DIM)

        # --- Global Attention ---
        self.global_attention = GlobalAttention(config.INPUT_DIM)

        # --- Projection to CRF emission scores ---
        self.hidden2tag = nn.Linear(config.INPUT_DIM, NUM_LABELS)

        # --- CRF ---
        self.crf = CRF(NUM_LABELS)

    def _build_input_representation(self, word_ids, char_ids, char_lengths):
        """
        Build E = [e_c; e_w] input representation.

        Returns:
            E: [batch, seq_len, input_dim]
            e_w: [batch, seq_len, word_embed_dim] (for GAT)
        """
        # Character representation
        e_c = self.char_bilstm(char_ids, char_lengths)  # [batch, seq_len, 2*char_hidden]

        # Word representation
        e_w = self.word_embed(word_ids)  # [batch, seq_len, word_embed_dim]

        # Concatenate
        E = torch.cat([e_c, e_w], dim=-1)  # [batch, seq_len, input_dim]
        E = self.input_dropout(E)

        return E, e_w

    def _apply_gat_batch(self, e_w, adj_matrices, mask):
        """Apply PII-GAT to each sequence in the batch."""
        batch_size, seq_len, _ = e_w.shape
        G_batch = torch.zeros(batch_size, seq_len, self.config.INPUT_DIM,
                              device=e_w.device)

        for i in range(batch_size):
            length = int(mask[i].sum().item())
            if length > 0:
                # Get valid portion
                e_w_i = e_w[i, :length]  # [length, word_embed_dim]
                adj_i = adj_matrices[i, :length, :length]  # [length, length]

                # Apply GAT
                g_i = self.pii_gat(e_w_i, adj_i)  # [length, input_dim]
                G_batch[i, :length] = g_i

        return G_batch

    def forward(self, word_ids, char_ids, char_lengths, adj_matrices,
                mask, labels=None):
        """
        Forward pass.

        Args:
            word_ids: [batch, seq_len]
            char_ids: [batch, seq_len, max_word_len]
            char_lengths: [batch, seq_len]
            adj_matrices: [batch, seq_len, seq_len]
            mask: [batch, seq_len] (1=valid, 0=padding)
            labels: [batch, seq_len] (optional, for training)

        Returns:
            dict with 'loss' and/or 'predictions'
        """
        # 1. Input Representation: E = [e_c; e_w]
        E, e_w = self._build_input_representation(word_ids, char_ids, char_lengths)

        # 2. Self Attention (Transformer): E -> S
        # Create padding mask for transformer (True = ignore)
        src_key_padding_mask = (mask == 0)
        S = self.transformer(E, src_key_padding_mask=src_key_padding_mask)

        # 3. PII-GAT: e_w + dependency graph -> G
        G = self._apply_gat_batch(e_w, adj_matrices, mask)

        # 4. Highway Layer: fuse S and G -> u
        u = self.highway(S, G)

        # 5. Gating Mechanism: combine E and u -> v
        v = self.gating(E, u)

        # 6. Global Attention: v -> R (pass mask for proper padding)
        R = self.global_attention(v, mask=mask)

        # 7. Projection to emission scores
        emissions = self.hidden2tag(R)  # [batch, seq_len, num_labels]

        result = {}

        if labels is not None:
            # Training: compute CRF loss
            loss = self.crf(emissions, labels, mask)
            result['loss'] = loss

        # Decoding
        predictions = self.crf.decode(emissions, mask)
        result['predictions'] = predictions

        return result


# =============================================================================
# Dependency Graph Construction
# =============================================================================

def build_dependency_graph(tokens):
    """
    Build dependency adjacency matrix from tokens using spaCy.

    Per paper (Section 5.2.1):
      "a sentence is represented as T = (t_1, t_2, ..., t_N), where nodes
       correspond to words and edges (edge_{k,i}) denote syntactic dependencies."

    Implementation details:
      - Edges are DIRECTED (child -> head), preserving the natural direction
        of syntactic dependencies (e.g., "his" -> "number" -> "is").
      - Self-loops are added (each node connects to itself).
      - Normalization: symmetric D^{-1/2} A D^{-1/2} (standard for GNNs).

    Returns:
        adj: numpy array [N, N] adjacency matrix (directed, normalized)
    """
    if not SPACY_AVAILABLE:
        # Fallback: linear chain (bidirectional)
        n = len(tokens)
        adj = np.eye(n)
        for i in range(n - 1):
            adj[i][i + 1] = 1
            adj[i + 1][i] = 1
        # Symmetric normalization
        degree = adj.sum(axis=1)
        d_inv_sqrt = np.where(degree > 0, np.power(degree, -0.5), 0)
        D = np.diag(d_inv_sqrt)
        adj = D @ adj @ D
        return adj

    text = " ".join(tokens)
    doc = nlp(text)

    n_tokens = len(tokens)
    adj = np.zeros((n_tokens, n_tokens))

    # Self-connections
    for i in range(n_tokens):
        adj[i][i] = 1

    # Align spaCy tokens to our tokens
    alignment = _align_tokens(tokens, [t.text for t in doc])

    # Add dependency edges (DIRECTED: child -> head)
    for token in doc:
        child = alignment.get(token.i, -1)
        head = alignment.get(token.head.i, -1)
        if 0 <= child < n_tokens and 0 <= head < n_tokens and child != head:
            adj[child][head] = 1  # child -> head (directed)

    # Symmetric normalization: D^{-1/2} A D^{-1/2}
    degree = adj.sum(axis=1)
    d_inv_sqrt = np.where(degree > 0, np.power(degree, -0.5), 0)
    D = np.diag(d_inv_sqrt)
    adj = D @ adj @ D

    return adj


def _align_tokens(orig_tokens, spacy_tokens):
    """Align spaCy tokens to original tokens."""
    alignment = {}
    o_idx, s_idx = 0, 0

    while o_idx < len(orig_tokens) and s_idx < len(spacy_tokens):
        ot = orig_tokens[o_idx].lower()
        st = spacy_tokens[s_idx].lower()

        if ot == st or ot.startswith(st) or st.startswith(ot):
            alignment[s_idx] = o_idx
            o_idx += 1
            s_idx += 1
        elif len(ot) > len(st):
            s_idx += 1
        else:
            o_idx += 1

    return alignment


# =============================================================================
# Vocabulary and Data Processing
# =============================================================================

class Vocabulary:
    """Word and character vocabularies."""

    def __init__(self):
        self.word2id = {"<PAD>": 0, "<UNK>": 1}
        self.id2word = {0: "<PAD>", 1: "<UNK>"}
        self.char2id = {"<PAD>": 0, "<UNK>": 1}
        self.id2char = {0: "<PAD>", 1: "<UNK>"}
        self.word_counter = Counter()

    def build_from_data(self, datasets, min_freq=1):
        """Build vocabulary from list of datasets."""
        for data in datasets:
            for item in data:
                for token in item['tokens']:
                    self.word_counter[token.lower()] += 1
                    for ch in token:
                        if ch not in self.char2id:
                            idx = len(self.char2id)
                            self.char2id[ch] = idx
                            self.id2char[idx] = ch

        for word, count in self.word_counter.items():
            if count >= min_freq and word not in self.word2id:
                idx = len(self.word2id)
                self.word2id[word] = idx
                self.id2word[idx] = word

    def load_glove(self, glove_path, embed_dim=200):
        """Load GloVe embeddings and create embedding matrix."""
        embeddings = {}
        if os.path.exists(glove_path):
            print(f"Loading GloVe from {glove_path}...")
            with open(glove_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == embed_dim + 1:
                        word = parts[0]
                        vec = np.array(parts[1:], dtype=np.float32)
                        embeddings[word] = vec
            print(f"Loaded {len(embeddings)} GloVe vectors.")
        else:
            print(f"Warning: GloVe file not found at {glove_path}. Using random init.")

        # Build embedding matrix
        vocab_size = len(self.word2id)
        embed_matrix = np.random.uniform(-0.25, 0.25, (vocab_size, embed_dim))
        embed_matrix[0] = np.zeros(embed_dim)  # PAD

        found = 0
        for word, idx in self.word2id.items():
            if word in embeddings:
                embed_matrix[idx] = embeddings[word]
                found += 1

        print(f"GloVe coverage: {found}/{vocab_size} ({100*found/vocab_size:.1f}%)")
        return torch.FloatTensor(embed_matrix)

    @property
    def word_vocab_size(self):
        return len(self.word2id)

    @property
    def char_vocab_size(self):
        return len(self.char2id)


def normalize_tag(tag):
    """Normalize BIOES tag to standard form."""
    if tag == 'O':
        return tag

    if not any(tag.startswith(p) for p in ['B-', 'I-', 'E-', 'S-']):
        return 'O'

    prefix = tag[:2]
    pii_type = tag[2:]

    type_map = {
        'age': 'age', 'Age': 'age', 'AGE': 'age',
        'contact': 'contact', 'Contact': 'contact', 'CONTACT': 'contact',
        'date': 'date', 'Date': 'date', 'DATE': 'date',
        'id': 'ID', 'Id': 'ID', 'ID': 'ID',
        'location': 'location', 'Location': 'location', 'LOCATION': 'location',
        'name': 'name', 'Name': 'name', 'NAME': 'name',
        'profession': 'profession', 'Profession': 'profession', 'PROFESSION': 'profession',
        'occupation': 'profession',  # alias
    }

    normalized = type_map.get(pii_type, pii_type)
    if normalized in Config.PII_TYPES:
        return f"{prefix}{normalized}"
    return 'O'


# =============================================================================
# Dataset
# =============================================================================

class PIIDataset(Dataset):
    """Dataset for MCA-PIIE training."""

    def __init__(self, data, vocab, max_seq_length=128, max_word_length=30):
        self.data = data
        self.vocab = vocab
        self.max_seq_length = max_seq_length
        self.max_word_length = max_word_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item['tokens'][:self.max_seq_length]
        labels = item['labels'][:self.max_seq_length]
        seq_len = len(tokens)

        # Word IDs
        word_ids = [self.vocab.word2id.get(t.lower(), 1) for t in tokens]

        # Character IDs
        char_ids = []
        char_lengths = []
        for token in tokens:
            chars = [self.vocab.char2id.get(c, 1) for c in token[:self.max_word_length]]
            char_lengths.append(len(chars))
            chars += [0] * (self.max_word_length - len(chars))
            char_ids.append(chars)

        # Label IDs
        label_ids = [LABEL2ID.get(l, 0) for l in labels]

        # Dependency graph
        try:
            adj = build_dependency_graph(tokens)
        except Exception:
            adj = np.eye(seq_len)

        # Pad everything to max_seq_length
        pad_len = self.max_seq_length - seq_len

        word_ids += [0] * pad_len
        char_ids += [[0] * self.max_word_length] * pad_len
        char_lengths += [0] * pad_len
        label_ids += [0] * pad_len

        # Pad adjacency matrix
        adj_padded = np.zeros((self.max_seq_length, self.max_seq_length))
        adj_padded[:seq_len, :seq_len] = adj[:seq_len, :seq_len]

        # Mask
        mask = [1.0] * seq_len + [0.0] * pad_len

        return {
            'word_ids': torch.tensor(word_ids, dtype=torch.long),
            'char_ids': torch.tensor(char_ids, dtype=torch.long),
            'char_lengths': torch.tensor(char_lengths, dtype=torch.long),
            'adj_matrices': torch.tensor(adj_padded, dtype=torch.float),
            'mask': torch.tensor(mask, dtype=torch.float),
            'labels': torch.tensor(label_ids, dtype=torch.long),
        }


# =============================================================================
# Data Loading Utilities
# =============================================================================

def load_csv_data(filepath, encoding='utf-8'):
    """Load PII data from CSV."""
    encodings = ['utf-8', 'latin1', 'cp1252', 'gbk', 'utf-16']
    if encoding not in encodings:
        encodings.insert(0, encoding)

    for enc in encodings:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            print(f"Loaded {len(df)} rows from {filepath} ({enc})")
            return df
        except Exception:
            continue

    raise RuntimeError(f"Cannot load {filepath}")


def process_bioes_data(df, text_col='Tweet Content', tag_col='Word_Level_BIOES'):
    """Process DataFrame with BIOES-tagged data."""
    processed = []

    # Prefer 'Tokens' column (pre-tokenized, aligned with tags)
    # Fall back to text_col if 'Tokens' not available
    has_tokens_col = 'Tokens' in df.columns

    for idx, row in df.iterrows():
        tags_str = row.get(tag_col, '')

        if pd.isna(tags_str) or str(tags_str).strip() == '':
            continue

        tags = str(tags_str).split()

        # Use pre-tokenized column if available (ensures alignment with tags)
        if has_tokens_col and not pd.isna(row.get('Tokens', None)):
            tokens = str(row['Tokens']).split()
        else:
            text = row.get(text_col, '')
            if pd.isna(text) or str(text).strip() == '':
                continue
            tokens = str(text).split()

        # Align lengths (handle minor mismatches)
        min_len = min(len(tokens), len(tags))
        if min_len == 0:
            continue
        tokens = tokens[:min_len]
        tags = tags[:min_len]

        # Normalize tags
        normalized_tags = [normalize_tag(t) for t in tags]

        if len(tokens) > 0:
            processed.append({
                'tokens': tokens,
                'labels': normalized_tags,
                'id': row.get('Tweet Id', f'row_{idx}'),
            })

    print(f"Processed {len(processed)} samples")
    return processed


def load_conll_format(filepath, token_col=0, tag_col=-1, separator=None):
    """Load data in CoNLL column format (one token per line, blank lines separate sentences)."""
    processed = []
    tokens, tags = [], []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '' or line.startswith('-DOCSTART-'):
                if tokens:
                    processed.append({
                        'tokens': tokens,
                        'labels': [normalize_tag(t) for t in tags],
                        'id': f'sent_{len(processed)}'
                    })
                    tokens, tags = [], []
            else:
                parts = line.split(separator)
                if len(parts) > max(token_col, tag_col if tag_col >= 0 else len(parts) + tag_col):
                    tokens.append(parts[token_col])
                    tags.append(parts[tag_col])

    if tokens:
        processed.append({
            'tokens': tokens,
            'labels': [normalize_tag(t) for t in tags],
            'id': f'sent_{len(processed)}'
        })

    print(f"Loaded {len(processed)} sentences from {filepath}")
    return processed


# =============================================================================
# Training Loop
# =============================================================================

def evaluate(model, dataloader, device):
    """Evaluate model and return metrics."""
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}

            result = model(
                word_ids=batch['word_ids'],
                char_ids=batch['char_ids'],
                char_lengths=batch['char_lengths'],
                adj_matrices=batch['adj_matrices'],
                mask=batch['mask'],
                labels=batch['labels']
            )

            total_loss += result['loss'].item()
            n_batches += 1

            predictions = result['predictions']
            labels = batch['labels'].cpu().numpy()
            masks = batch['mask'].cpu().numpy()

            for pred, label, m in zip(predictions, labels, masks):
                length = int(m.sum())
                for j in range(length):
                    all_preds.append(ID2LABEL.get(pred[j], 'O'))
                    all_labels.append(ID2LABEL.get(label[j], 'O'))

    # Compute metrics
    # Binary: PII vs non-PII
    binary_true = [0 if l == 'O' else 1 for l in all_labels]
    binary_pred = [0 if p == 'O' else 1 for p in all_preds]

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        binary_true, binary_pred, average='binary', zero_division=0
    )

    # Per-category F1
    category_f1 = {}
    for pii_type in Config.PII_TYPES:
        cat_true = [1 if pii_type in l else 0 for l in all_labels]
        cat_pred = [1 if pii_type in p else 0 for p in all_preds]
        if sum(cat_true) > 0:
            _, _, cf1, _ = precision_recall_fscore_support(
                cat_true, cat_pred, average='binary', zero_division=0
            )
            category_f1[pii_type] = cf1

    avg_loss = total_loss / max(n_batches, 1)

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'category_f1': category_f1
    }


def train_model(model, train_loader, eval_loader, config, save_dir='./mca_piie_model'):
    """Train MCA-PIIE with early stopping."""
    device = config.DEVICE
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )

    os.makedirs(save_dir, exist_ok=True)

    best_f1 = 0
    patience_counter = 0

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_loss = 0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}", leave=True)
        for batch_idx, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()

            result = model(
                word_ids=batch['word_ids'],
                char_ids=batch['char_ids'],
                char_lengths=batch['char_lengths'],
                adj_matrices=batch['adj_matrices'],
                mask=batch['mask'],
                labels=batch['labels']
            )

            loss = result['loss']
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            # Update progress bar with running loss
            pbar.set_postfix(loss=f"{total_loss/n_batches:.4f}")

        # Evaluate
        metrics = evaluate(model, eval_loader, device)
        scheduler.step(metrics['f1'])

        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        print(f"  Train Loss: {total_loss/n_batches:.4f}")
        print(f"  Eval  Loss: {metrics['loss']:.4f}")
        print(f"  Accuracy:   {metrics['accuracy']:.4f}")
        print(f"  Precision:  {metrics['precision']:.4f}")
        print(f"  Recall:     {metrics['recall']:.4f}")
        print(f"  F1:         {metrics['f1']:.4f}")
        if metrics['category_f1']:
            print(f"  Category F1: {metrics['category_f1']}")

        # Early stopping
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))
            print(f"  -> New best F1: {best_f1:.4f}, model saved.")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}. Best F1: {best_f1:.4f}")
                break

    # Load best model
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pt')))
    return model


# =============================================================================
# Deep Transfer Learning (per Section 5.2 & Figure 4)
# =============================================================================

def transfer_learning(source_model, target_model):
    """
    Transfer specific layers from source model to target model.

    Per paper (Figure 4, highlighted in green):
      Transferred components:
        - Character Bi-LSTM
        - PII-GAT
        - Transformer (Self Attention)
        - Global Attention

      Fine-tuned on target domain:
        - Input layer (word embeddings)
        - Output layer (CRF)
    """
    print("Transferring layers from source to target model...")

    # Transfer Character Bi-LSTM
    target_model.char_bilstm.load_state_dict(
        source_model.char_bilstm.state_dict()
    )

    # Transfer PII-GAT
    target_model.pii_gat.load_state_dict(
        source_model.pii_gat.state_dict()
    )

    # Transfer Transformer (Self Attention)
    target_model.transformer.load_state_dict(
        source_model.transformer.state_dict()
    )

    # Transfer Global Attention
    target_model.global_attention.load_state_dict(
        source_model.global_attention.state_dict()
    )

    # Also transfer Highway and Gating (these are part of the attention pipeline)
    target_model.highway.load_state_dict(
        source_model.highway.state_dict()
    )
    target_model.gating.load_state_dict(
        source_model.gating.state_dict()
    )

    print("Transfer complete. Fine-tune on target domain data.")
    return target_model


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    """
    MCA-PIIE pipeline with mode selection.

    Modes (corresponding to paper experiments):
      --mode full_dtl       Full pipeline: Phase 1 (source) + Phase 2 (transfer + fine-tune target)
                            -> This is the main MCA-PIIE result (Table 8, target domain)

      --mode target_only    Train directly on target domain, NO transfer learning
                            -> Corresponds to "MCA-PIIE w/o DTL" in ablation (Table 9)

      --mode source_only    Train and evaluate on source domain only
                            -> Corresponds to Table 8, source domain results

      --mode ablation       Run all ablation variants (w/o GAT, w/o Transformer, etc.)
                            -> Corresponds to Table 9
    """
    parser = argparse.ArgumentParser(
        description="MCA-PIIE: Multi-Context Attention for PII Extraction",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--mode', type=str, default='full_dtl',
                        choices=['full_dtl', 'target_only', 'source_only', 'ablation'],
                        help="""Training mode:
  full_dtl     - Full pipeline with Deep Transfer Learning (default)
                 Phase 1: train on source domain
                 Phase 2: transfer + fine-tune on target domain
  target_only  - Train directly on target domain (no DTL)
                 Corresponds to "w/o DTL" ablation in Table 9
  source_only  - Train and evaluate on source domain only
                 Corresponds to source domain results in Table 8
  ablation     - Run all ablation experiments (Table 9)""")
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--save_dir', type=str, default='./mca_piie_output',
                        help='Directory to save models and results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--max_source', type=int, default=None,
                        help='Override max source samples (default: 50000)')

    args = parser.parse_args()

    # ---- Setup ----
    config = Config()
    if args.epochs:
        config.NUM_EPOCHS = args.epochs
        config.SOURCE_EPOCHS = args.epochs  # also cap source
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.lr:
        config.LEARNING_RATE = args.lr
    if args.max_source:
        config.MAX_SOURCE_SAMPLES = args.max_source

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 60)
    print("MCA-PIIE: Multi-Context Attention for PII Extraction")
    print("=" * 60)
    print(f"  Mode:       {args.mode}")
    print(f"  Device:     {config.DEVICE}")
    print(f"  Labels:     {NUM_LABELS} ({len(Config.PII_TYPES)} PII types)")
    print(f"  Epochs:     {config.NUM_EPOCHS}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  LR:         {config.LEARNING_RATE}")
    print(f"  Save dir:   {args.save_dir}")
    print("=" * 60)

    # =================================================================
    # DATA PATHS — Edit these to match your local file locations
    # =================================================================

    # --- Source Domain (6 public PII datasets, per Table 5) ---
    # Note: i2b2 2014 replaced with PIILO (Kaggle PII Detection) because
    # n2c2 data portal is temporarily unavailable. PIILO provides ID_NUM
    # coverage that other datasets lack.
    SOURCE_FILES = {
        'conll_2003':      ('../data/source/conll2003_all.csv',      'csv'),
        'i2b2_2014':       ('../data/source/i2b2_2014.csv',          'csv'),
        'gmb':             ('../data/source/gmb.csv',                'csv'),
        'wnut17':          ('../data/source/wnut17.csv',             'csv'),
        'broad_twitter':   ('../data/source/broad_twitter.csv',      'csv'),
        'resume_ner':      ('../data/source/resume_ner.csv',         'csv'),
    }

    # --- Target Domain (self-collected Twitter data) ---
    # NOTE: Original PII_tweet.csv cannot be publicly released due to privacy
    # considerations and IRB restrictions. A small de-identified sample is
    # provided in ../data/sample/sample_pii_tweets.csv for pipeline verification.
    TARGET_FILE = '../data/sample/sample_pii_tweets.csv'

    # --- GloVe embeddings ---
    # Download GloVe-Twitter-200d from https://nlp.stanford.edu/projects/glove/
    # and place glove.twitter.27B.200d.txt in ../embeddings/
    config.GLOVE_FILE = '../embeddings/glove.twitter.27B.200d.txt'

    # =================================================================
    # Load data based on mode
    # =================================================================

    source_data = []
    target_train = []
    target_test = []

    need_source = args.mode in ('full_dtl', 'source_only', 'ablation')
    need_target = args.mode in ('full_dtl', 'target_only', 'ablation')

    if need_source:
        print("\n[Loading source domain data]")
        for name, (path, fmt) in SOURCE_FILES.items():
            if not os.path.exists(path):
                print(f"  [SKIP] {name}: {path} not found")
                continue
            if fmt == 'csv':
                df = load_csv_data(path)
                data = process_bioes_data(df)
            else:
                data = load_conll_format(path)
            source_data.extend(data)
            print(f"  [OK]   {name}: {len(data)} sentences")
        print(f"  Total source: {len(source_data)} sentences")
        if not source_data:
            print("\nERROR: Mode '{}' requires source-domain datasets, but none were "
                  "found under ../data/source/.".format(args.mode))
            print("       See ../data/source/README.md for download instructions for "
                  "CoNLL-2003, GMB, WNUT-17, Broad Twitter, Resume-NER, and i2b2.")
            print("       To run on the target sample only, use: --mode target_only")
            return

    if need_target:
        print("\n[Loading target domain data]")
        if os.path.exists(TARGET_FILE):
            df = load_csv_data(TARGET_FILE)
            target_data = process_bioes_data(df)
            random.seed(args.seed)
            random.shuffle(target_data)
            print(f"  Target: {len(target_data)} sentences")
        else:
            print(f"  WARNING: {TARGET_FILE} not found. Using dummy data.")
            target_data = [{
                'tokens': ['My', 'name', 'is', 'John', 'Smith'],
                'labels': ['O', 'O', 'O', 'B-name', 'E-name'],
                'id': 'demo_1'
            }] * 10

    # ---- Build Vocabulary ----
    print("\n[Building vocabulary]")
    vocab = Vocabulary()
    all_data = source_data + (target_data if need_target else [])
    if not all_data:
        print("ERROR: No data loaded. Check your file paths.")
        return
    vocab.build_from_data([all_data])
    print(f"  Words: {vocab.word_vocab_size}, Chars: {vocab.char_vocab_size}")

    glove_embeddings = vocab.load_glove(config.GLOVE_FILE, config.WORD_EMBED_DIM)

    # =================================================================
    # Run selected mode (all modes use 5-fold CV internally)
    # =================================================================

    if args.mode == 'source_only':
        _run_source_only(config, vocab, glove_embeddings, source_data, args.save_dir)

    elif args.mode == 'target_only':
        _run_target_only(config, vocab, glove_embeddings, target_data, args.save_dir)

    elif args.mode == 'full_dtl':
        _run_full_dtl(config, vocab, glove_embeddings, source_data, target_data, args.save_dir)

    elif args.mode == 'ablation':
        _run_ablation(config, vocab, glove_embeddings, source_data, target_data, args.save_dir)

    print("\nAll done.")


# =============================================================================
# 5-Fold Cross-Validation utility
# =============================================================================

def kfold_split(data, k=5):
    """Split data into k folds. Returns list of (train, test) tuples."""
    fold_size = len(data) // k
    folds = []
    for i in range(k):
        test_start = i * fold_size
        test_end = test_start + fold_size if i < k - 1 else len(data)
        test_data = data[test_start:test_end]
        train_data = data[:test_start] + data[test_end:]
        folds.append((train_data, test_data))
    return folds


def run_kfold_experiment(config, vocab, glove_embeddings, data, save_dir,
                         label, model_factory_fn):
    """
    Run 5-fold CV for a given model factory function.

    Per paper Section 5.3:
      "For all experiments, 5-fold cross-validation was adopted.
       Paired t-tests were used to identify statistically significant
       differences between performance metrics."

    Args:
        model_factory_fn: callable() -> nn.Module (creates a fresh model each fold)

    Returns:
        dict with mean and std of all metrics across folds
    """
    k = config.K_FOLDS
    folds = kfold_split(data, k)

    all_metrics = []

    for fold_idx, (train_data, test_data) in enumerate(folds):
        print(f"\n{'─'*50}")
        print(f"  {label} — Fold {fold_idx+1}/{k}  "
              f"(train={len(train_data)}, test={len(test_data)})")
        print(f"{'─'*50}")

        train_loader = DataLoader(
            PIIDataset(train_data, vocab, config.MAX_SEQ_LENGTH),
            batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
        )
        test_loader = DataLoader(
            PIIDataset(test_data, vocab, config.MAX_SEQ_LENGTH),
            batch_size=config.BATCH_SIZE
        )

        # Fresh model each fold
        model = model_factory_fn()

        fold_save = os.path.join(save_dir, f'fold_{fold_idx+1}')
        model = train_model(model, train_loader, test_loader, config, save_dir=fold_save)

        metrics = evaluate(model, test_loader, config.DEVICE)
        metrics['fold'] = fold_idx + 1
        all_metrics.append(metrics)

        print(f"  Fold {fold_idx+1} -> "
              f"Acc={metrics['accuracy']:.4f}  "
              f"P={metrics['precision']:.4f}  "
              f"R={metrics['recall']:.4f}  "
              f"F1={metrics['f1']:.4f}")

    # Aggregate across folds
    _print_cv_summary(all_metrics, label)
    return all_metrics


def run_kfold_dtl_experiment(config, vocab, glove_embeddings, source_data,
                             target_data, save_dir, label):
    """
    Run 5-fold CV with Deep Transfer Learning.
    Phase 1 (source training) is done ONCE; Phase 2 (fine-tune) is per fold.

    This matches the paper's setup: source model is shared, target evaluation
    is 5-fold CV.
    """
    k = config.K_FOLDS

    # Phase 1: Train source model ONCE on all source data
    print("\n--- Phase 1: Source Domain Training (shared across folds) ---")

    # Subsample source data if too large (preserving PII ratio)
    if config.MAX_SOURCE_SAMPLES and len(source_data) > config.MAX_SOURCE_SAMPLES:
        print(f"  Subsampling source: {len(source_data)} -> {config.MAX_SOURCE_SAMPLES}")
        pii_sents = [s for s in source_data if any(t != 'O' for t in s['labels'])]
        non_pii_sents = [s for s in source_data if all(t == 'O' for t in s['labels'])]
        random.shuffle(pii_sents)
        random.shuffle(non_pii_sents)
        # Keep all PII sentences (they're the minority), fill rest with non-PII
        if len(pii_sents) >= config.MAX_SOURCE_SAMPLES:
            sampled = pii_sents[:config.MAX_SOURCE_SAMPLES]
        else:
            remaining = config.MAX_SOURCE_SAMPLES - len(pii_sents)
            sampled = pii_sents + non_pii_sents[:remaining]
        random.shuffle(sampled)
        source_data_used = sampled
        print(f"  PII sentences kept: {len(pii_sents)} / Non-PII sampled: {len(sampled) - len(pii_sents)}")
    else:
        source_data_used = source_data

    source_split = int(len(source_data_used) * 0.9)
    source_train = source_data_used[:source_split]
    source_eval = source_data_used[source_split:]

    source_train_loader = DataLoader(
        PIIDataset(source_train, vocab, config.MAX_SEQ_LENGTH),
        batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
    )
    source_eval_loader = DataLoader(
        PIIDataset(source_eval, vocab, config.MAX_SEQ_LENGTH),
        batch_size=config.BATCH_SIZE
    )

    source_model = MCAPIIE(config=config, char_vocab_size=vocab.char_vocab_size,
                           word_vocab_size=vocab.word_vocab_size,
                           pretrained_word_embeddings=glove_embeddings)

    source_lr = config.LEARNING_RATE
    # Use fewer epochs for source (larger dataset converges faster)
    original_epochs = config.NUM_EPOCHS
    config.NUM_EPOCHS = config.SOURCE_EPOCHS
    source_model = train_model(source_model, source_train_loader, source_eval_loader,
                               config, save_dir=os.path.join(save_dir, 'source_phase'))
    config.NUM_EPOCHS = original_epochs  # Restore for target fine-tuning

    # Phase 2: 5-fold CV on target domain with transfer
    print(f"\n--- Phase 2: {k}-Fold CV on Target Domain (with DTL) ---")
    folds = kfold_split(target_data, k)
    all_metrics = []

    config.LEARNING_RATE = source_lr * 0.5  # Lower LR for fine-tuning

    for fold_idx, (train_data, test_data) in enumerate(folds):
        print(f"\n{'─'*50}")
        print(f"  {label} — Fold {fold_idx+1}/{k}  "
              f"(train={len(train_data)}, test={len(test_data)})")
        print(f"{'─'*50}")

        train_loader = DataLoader(
            PIIDataset(train_data, vocab, config.MAX_SEQ_LENGTH),
            batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
        )
        test_loader = DataLoader(
            PIIDataset(test_data, vocab, config.MAX_SEQ_LENGTH),
            batch_size=config.BATCH_SIZE
        )

        # Fresh target model + transfer from source each fold
        target_model = MCAPIIE(config=config, char_vocab_size=vocab.char_vocab_size,
                               word_vocab_size=vocab.word_vocab_size,
                               pretrained_word_embeddings=glove_embeddings)
        target_model = transfer_learning(source_model, target_model)

        fold_save = os.path.join(save_dir, f'target_fold_{fold_idx+1}')
        target_model = train_model(target_model, train_loader, test_loader,
                                   config, save_dir=fold_save)

        metrics = evaluate(target_model, test_loader, config.DEVICE)
        metrics['fold'] = fold_idx + 1
        all_metrics.append(metrics)

        print(f"  Fold {fold_idx+1} -> "
              f"Acc={metrics['accuracy']:.4f}  "
              f"P={metrics['precision']:.4f}  "
              f"R={metrics['recall']:.4f}  "
              f"F1={metrics['f1']:.4f}")

    config.LEARNING_RATE = source_lr  # Restore

    _print_cv_summary(all_metrics, label)
    return all_metrics


def _print_cv_summary(all_metrics, label):
    """Print mean ± std across folds (matches paper's reported format)."""
    import numpy as np

    print(f"\n{'═'*60}")
    print(f"  {label} — {len(all_metrics)}-Fold CV Summary")
    print(f"{'═'*60}")

    for key in ['accuracy', 'precision', 'recall', 'f1']:
        values = [m[key] for m in all_metrics]
        mean = np.mean(values)
        std = np.std(values)
        print(f"  {key:12s}: {mean:.4f} ± {std:.4f}  "
              f"(folds: {', '.join(f'{v:.4f}' for v in values)})")

    # Per-category F1 summary
    all_cats = set()
    for m in all_metrics:
        all_cats.update(m.get('category_f1', {}).keys())

    if all_cats:
        print(f"\n  Per-category F1:")
        for cat in sorted(all_cats):
            values = [m.get('category_f1', {}).get(cat, 0) for m in all_metrics]
            mean = np.mean(values)
            std = np.std(values)
            print(f"    {cat:12s}: {mean:.4f} ± {std:.4f}")

    print(f"{'═'*60}")


# =============================================================================
# Mode: source_only (Table 8, source domain)
# =============================================================================

def _run_source_only(config, vocab, glove_embeddings, source_data, save_dir):
    print("\n" + "=" * 60)
    print(f"MODE: source_only — {config.K_FOLDS}-Fold CV on source domain")
    print("=" * 60)

    def model_factory():
        return MCAPIIE(config=config, char_vocab_size=vocab.char_vocab_size,
                       word_vocab_size=vocab.word_vocab_size,
                       pretrained_word_embeddings=glove_embeddings)

    run_kfold_experiment(config, vocab, glove_embeddings, source_data,
                         os.path.join(save_dir, 'source_only'),
                         "Source Domain", model_factory)


# =============================================================================
# Mode: target_only — w/o DTL (Table 9 ablation)
# =============================================================================

def _run_target_only(config, vocab, glove_embeddings, target_data, save_dir):
    print("\n" + "=" * 60)
    print(f"MODE: target_only — {config.K_FOLDS}-Fold CV on target (w/o DTL)")
    print("       (corresponds to 'MCA-PIIE w/o DTL' in Table 9)")
    print("=" * 60)

    def model_factory():
        return MCAPIIE(config=config, char_vocab_size=vocab.char_vocab_size,
                       word_vocab_size=vocab.word_vocab_size,
                       pretrained_word_embeddings=glove_embeddings)

    run_kfold_experiment(config, vocab, glove_embeddings, target_data,
                         os.path.join(save_dir, 'target_only'),
                         "Target Domain (w/o DTL)", model_factory)


# =============================================================================
# Mode: full_dtl — Full pipeline (Table 8, target domain)
# =============================================================================

def _run_full_dtl(config, vocab, glove_embeddings, source_data, target_data, save_dir):
    print("\n" + "=" * 60)
    print(f"MODE: full_dtl — Source training + {config.K_FOLDS}-Fold CV on target (with DTL)")
    print("       (corresponds to MCA-PIIE in Table 8, target domain)")
    print("=" * 60)

    run_kfold_dtl_experiment(config, vocab, glove_embeddings, source_data,
                             target_data, os.path.join(save_dir, 'full_dtl'),
                             "MCA-PIIE (full DTL)")


# =============================================================================
# Mode: ablation — All ablation variants (Table 9)
# =============================================================================

def _run_ablation(config, vocab, glove_embeddings, source_data, target_data, save_dir):
    print("\n" + "=" * 60)
    print(f"MODE: ablation — All Table 9 variants with {config.K_FOLDS}-Fold CV")
    print("=" * 60)

    # 1. Full MCA-PIIE (with DTL)
    print("\n[1/5] Full MCA-PIIE (with DTL)...")
    run_kfold_dtl_experiment(config, vocab, glove_embeddings, source_data,
                             target_data, os.path.join(save_dir, 'ablation_full'),
                             "MCA-PIIE (full)")

    # 2. w/o DTL
    print("\n[2/5] MCA-PIIE w/o DTL...")
    def factory_no_dtl():
        return MCAPIIE(config=config, char_vocab_size=vocab.char_vocab_size,
                       word_vocab_size=vocab.word_vocab_size,
                       pretrained_word_embeddings=glove_embeddings)
    run_kfold_experiment(config, vocab, glove_embeddings, target_data,
                         os.path.join(save_dir, 'ablation_no_dtl'),
                         "MCA-PIIE (w/o DTL)", factory_no_dtl)

    # 3. w/o PII-GAT + Dependency Graph
    print("\n[3/5] MCA-PIIE w/o PII-GAT...")
    def factory_no_gat():
        model = MCAPIIE(config=config, char_vocab_size=vocab.char_vocab_size,
                        word_vocab_size=vocab.word_vocab_size,
                        pretrained_word_embeddings=glove_embeddings)
        # Disable GAT: freeze and zero out
        for param in model.pii_gat.parameters():
            param.requires_grad = False
            param.data.zero_()
        return model
    run_kfold_experiment(config, vocab, glove_embeddings, target_data,
                         os.path.join(save_dir, 'ablation_no_gat'),
                         "MCA-PIIE (w/o PII-GAT)", factory_no_gat)

    # 4. w/o Transformer (Self Attention)
    print("\n[4/5] MCA-PIIE w/o Transformer...")
    def factory_no_trans():
        model = MCAPIIE(config=config, char_vocab_size=vocab.char_vocab_size,
                        word_vocab_size=vocab.word_vocab_size,
                        pretrained_word_embeddings=glove_embeddings)
        model.transformer = nn.Identity()
        return model
    run_kfold_experiment(config, vocab, glove_embeddings, target_data,
                         os.path.join(save_dir, 'ablation_no_trans'),
                         "MCA-PIIE (w/o Transformer)", factory_no_trans)

    # 5. w/o Global Attention
    print("\n[5/5] MCA-PIIE w/o Global Attention...")
    def factory_no_global():
        model = MCAPIIE(config=config, char_vocab_size=vocab.char_vocab_size,
                        word_vocab_size=vocab.word_vocab_size,
                        pretrained_word_embeddings=glove_embeddings)
        model.global_attention = nn.Identity()
        return model
    run_kfold_experiment(config, vocab, glove_embeddings, target_data,
                         os.path.join(save_dir, 'ablation_no_global'),
                         "MCA-PIIE (w/o Global Attention)", factory_no_global)

    print("\n" + "=" * 60)
    print("Ablation complete. Compare CV summaries above with Table 9.")
    print("=" * 60)


# =============================================================================
# Helper: Print final evaluation metrics
# =============================================================================

if __name__ == "__main__":
    main()