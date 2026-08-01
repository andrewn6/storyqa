from __future__ import annotations 
from configparser import ConverterMapping
import os, re, math, json, argparse 
from dataclasses import dataclass
from this import d 

import numpy as np 
from tinygrad import Tensor, dtypes, Context
from tinygrad.nn import Linear, LayerNorm, Embedding 
from tinygrad.nn.optim import AdamW 
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict, safe_save, safe_load 

CTX = 256 
D_MODEL = 192 
N_LAYERS = 6 
N_HEADS = 6 
D_FF = 1024 
PAD_IDX, CLS_IDX, SEP_IDX, MASK_IDX = 0, 1, 2, 3
SPECIAL = ["[pad]", "[cls]", "[sep]", "[mask]"]

PEOPLE = ["andrew", "marc", "sean", "bob"]
OBJECTS = ["key", "ball", "book", "coin"]
CONTAINERS = ["box", "basket"]
LOCATIONS = ["garden", "kitchen", "office", "bedroom", "hall"]
TEMPLATE = ["moved", "to", "picked", "up", "put", "down", "gave", "in", "from",
            "took", "where", "is", "who", "has", "what", "does", "have", "the"]
PUNCT = [".", "?", ","]

"""
Locations + people + objects + containers + nobody 
"""
ANSWER_TOKENS = sorted(set(LOCATIONS + PEOPLE + OBJECTS + CONTAINERS + ["nobody", "nothing"]))
ANS2ID = {t: i for i, t in enumerate(ANSWER_TOKENS)}
N_CLASSES = len(ANSWER_TOKENS)

"""
Full vocabulary
"""
VOCAB = SPECIAL + sorted(set(TEMPLATE + PUNCT + PEOPLE + OBJECTS + CONTAINERS + LOCATIONS + ["nobody", "nothing"]))
TOK2IDS = {t: i for i, t in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)

CKPT_DIR = "ckpts"
CKPT_PATH = os.path.join(CKPT_DIR, "storyqa.safetensors")
META_PATH = os.path.join(CKPT_DIR, "storyqa.json")

@dataclass 
class World:
    person_loc: dict[str,str] 
    item_loc: dict[str,str] 
    holder: dict[str,str]
    obj_container: dict[str,str]

    @staticmethod 
    def random(rng: np.random.Generator) -> "World":
        items = OBJECTS + CONTAINERS 
        return World(
                person_loc={p: rng.choice(LOCATIONS) for p in PEOPLE}, 
                item_loc={it: rng.choice(LOCATIONS) for it in items},
                holder={it: None for it in items},
                obj_container={o: None for o in OBJECTS}
        )

    def holds(self, person: str):
        for it in OBJECTS + CONTAINERS:
            if self.holder.get(it) == person:
                return it 
        return None 

    def contents(self, container: str):
        inside = [o for o in OBJECTS if self.obj_container.get(o) == container]
        return inside or None 

    def loc(self, x: str) -> str | None:
        if x in PEOPLE:
            return self.person_loc[x]
        owner = self.holder.get(x)
        if owner is not None:
            return self.person_loc[owner]
        if x in OBJECTS:
            c = self.obj_container.get(x) 
            if c is not None:
                return self.loc(c)
        return self.item_loc.get(x)


"""
difficulty: 0, short/direct, 1= distractors/overwrites, 2 = long multi-hop + transfers
"""
DIFF_LEN = {0: (1, 3), 1: (4, 8), 2:(8,16)}

def _valid_actions(world: World, rng: np.random.Generator, difficulty: int):
    acts = []
    for p in PEOPLE:
        acts.append(("move", p, rng.choice(LOCATIONS)))
    for p in PEOPLE:
        pl = world.loc(p)
        for it in OBJECTS + CONTAINERS:
            if world.holder.get(it) is None and world.loc(it) == pl and world.holds(p) is None:
                acts.append(("pickup", p, it))

    for p in PEOPLE:
        it = world.holds(p)
        if it is not None:
            acts.append(("drop", p, it))
    if difficulty >= 1:
        for p in PEOPLE:
            it = world.holds(p)
            if it is None:
                continue 
            for q in PEOPLE:
                if q != p and world.loc(q) == world.loc(p) and world.holds(q) is None:
                    acts.append(("give", p, q, it))
    if difficulty >= 2:
        for p in PEOPLE:
            pl = world.loc(p)
            it = world.holds(p)
            for c in CONTAINERS:
                if world.loc(c) == pl and world.contents(c) is None and it in OBJECTS:
                    acts.append(("put_in", p, it, c))
                if world.loc(c) == pl and worlds.holds(p) is None and world.contents(c):
                    acts.append(("take_from", p, world.contents(c)[0], c))
    return acts 

def _apply(world: World, act, rng: np.random.Generator) -> list[str]:
    kind = act[0]
    if kind == "move":
        _, p, loc = act 
        world.person_loc[p] = loc 
        return [p, "moved", "to", "the", loc, "."]
    if kind == "pickup":
        _, p, it = act 
        if it in OBJECTS and world.obj_container.get(it) is not None:
            world.obj_container[it] = None 
        world.holder[it] = p 
        return [p, "picked", "up", "the", it, "."]
    if kind == "drop":
        _, p, it = act 
        world.holder[it] = None 
        world.item_loc[it] = world.loc(p)
        return [p, "put", "down", "the", it, "."]
    if kind == "give":
        _ p, q, it = act 
        world.holder[it] = q
    if kind == "put_in":
        _, p, it, c = act
        world.holder[it] = None
        world.obj_container[it] = c
        return [p, "put", "the", it, "in", "the", c, "."]
    if kind == "take_from":
        _, p, it, c = act
        world.obj_container[it] = None
        world.holder[it] = p
        return [p, "took", "the", it, "from", "the", c, "."]
    raise ValueError(kind)

def generate_story(rng: np.random.Generator, difficulty: int):
    world = World.random(rng)
    lo, hi = DIFF_LEN[difficulty]
    n = int(rng.integers(lo, hi + 1))
    tokens: list[str] = [] 
    for _ in range(n):
        acts = _valid_actions(world, rng, difficulty)
        if not acts:
            break 
        tokens += _apply(world, acts[int(rng.integers(len(acts)))], rng)
    return world, tokens

def make_question(rng: np.random.Generator, world: World, difficulty: int):
    qs = []
    for p in PEOPLE:
        qs.append((["where", "is", p, "?"], world.loc(p)))
    for it in OBJECTS + CONTAINERS:
        qs.append((["where", "is", "the", it, "?"], world.loc(it)))
        holder = world.holder.get(it)
        qs.append((["who", "has", "the", it, "?"], holder if holder is not None else "nobody"))
    for p in PEOPLE:
        it = world.holds(p)
        qs.append((["what", "does", p, "have", "?"], it if it is not None else "nothing"))
    if difficulty >= 2:
        for c in CONTAINERS:
            inside = world.contents(c)
            qs.append((["what", "is", "in", "the", c, "?"], inside[0] if inside else "nothing"))
    return qs[int(rng.integers(len(qs)))]

"""
tokenization!
"""

_PUNCT = set(".?,")

def tokenize(text: str) -> list[str]:
    raw = text.lower().split()
    out: list[str] = []
    for w in raw:
        if w in SPECIAL:
            out.append(w)
            continue 
        if re.fullmatch(r"[.,?]+", w):
            out.append(w)
            continue
        if w and w[-1] in _PUNCT:
            out.append(w[:-1])
            out.append(w[-1])
        else:
            out.append(w)
    return out 

def detokenize(tokens: list[str]) -> str:
    out = ""
    for t in tokens:
        if t in _PUNCT:
            out = out.rstrip() + t + " "
        else:
            out += t + " "
    return out.strip()

def encode(story_tokens: list[str], question_tokens: list[str], answer: str):
    seq = ["[cls]"] + story_tokens + ["[sep]"] + question_tokens + ["[sep]"]
    seq = seq[:CTX]
    n_real = len(seq)
    ids = [TOK2IDS[t] for t in seq]
    seg = [0] * (len(story_tokens) + 2) [1] * (len(questions_tokens) + 1 )
    seg = seg[:n_real]
    if n_real < CTX:
        ids += [PAD_IDX] * (CTX - n_real)
        seg += [0] * (CTX - n_real)
    label = ANS2ID[answer]
    return ids, seg, label, seq 

def sample_difficulty(rng: np.random.Generator, progress: float) -> int:
    if progress < 0.25:
        w = [0.8, 0.2, 0.0]
    elif progress < 0.6:
        w = [0.3, 0.5, 0.2]
    else:
        w = [0.1, 0.4, 0.5]
    return int(rng.choice(3, p=w))

def make_batch(rng: np.random.Generator, n: int, difficulty: int | None = None, progress: float = 0.0):
    ids_b, seg_b, lab_b, seqs = [], [], [], []
    for _ in range(n):
        d = difficulty if difficulty is not None else sample_difficulty(rng, progress)
        world, story = generate_story(rng, d)
        qtoks, ans = make_question(rng, world, d)
        ids, seg, lab, seq = encode(story, qtoks, ans)
        ids_b.append(ids); seg_b.append(seg); lab_b.append(lab); seqs.append(seq)
    return (np.array(ids_b, dtype=np.int32),
            np.array(seg_b, dtype=np.int32),
            np.array(lab_b, dtype=np.int32),
            seqs)

class Attention:
   """
   Scaled dot product self-attention 

   My own notes:
   Every token emits three vectors via learned projections
   Q = X Wq, K = X Wk, V = X Wv

   Attention scores measure how much token i "queries" token j:
    scores[i, j] = (Q_i . K_j) / sqrt(d_head)
    
   We divide sqrt(d_head) so the dot products stay on a sensible scale;
   
   This is BIDIRECTIONAL (BERT-style): there is no causal mask, so every token
    attends to every other token in both directions. The only masking is over
    [pad] keys, which get a -1e9 score so they contribute 0 weight.

    Multi-head: d_model is split into H heads of d_head = d_model / H. Each head
    runs the above attention independently, letting the model attend to different
    relations in parallel. Heads are then concatenated and projected back to d_model.
   """  
   def __init__(self, d_model: int, n_heads: int):
       assert d_model % n_heads == 0 
       self.n_heads = n_heads
       self.d_head = d_model // n_heads
       self.q = Linear(d_model, d_model)
       self.k = Linear(d_model, d_model)
       self.v = Linear(d_model, d_model)
       self.o = Linear(d_model, d_model)
       self.last_attn = None  # stored for `inspect`

    def __call__(self, x: Tensor, key_mask: Tensor) -> Tensor:
        B, T, C = x.shape 
        H, Dh = self.n_heads, self.d_head
        # project to Q, K, V then split each token into H heads (B, T, C) -> (B, H, t, Dh)
        q = self.q(x).reshape(B, T, H, Dh).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, T, H, Dh).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, T, H, Dh).permute(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(Dh)
        scores = (q @ k.permute(0, 1, 3, 2)) * scale 
        scores = scores + key_mask 
        attn = scores.softmax(axis=-1) # softmax over K (keys)
        self.last_attn = attn 

        out = attn @ v # weighted sum of values, B, H, T, Dh
        out = out.permute(0, 2, 1, 3).reshape(B, T, C) # concat heads back to model
        return self.o(out)

class MLP:
    """
    Position-wise feed-forward sublayer: expand -> GELU -> project back.

    Notes for self: attention mixes information across tokens thru heads; MLP thinks on each tokens representation indiviudally. Position wise. The same tiny net is applied independently to every token, with no looking at neighbors.

    Two linear layers with a nonlinearity between them.

    Without the activation, two linear layesr would collapse into one. So we need a nonlinearity. GeLu is smooth ReLU it passes large positives thru, zeroes out large negatives, adn rounds the corner near zero unlike Relus sharpness. which trains better in transformers.
    """
    def __init__(self, d_model: int, d_ff: int):
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)

    def __call__(self, x: Tensor) -> Tensor:
        return self.fc2(self.fc1(x).gelu())

class Block:
    """
    One pre-norm transformer block: self attention + MLP in a residual

    x = x + Attn(layernorm(x)) -- self attention sublayer 
    x = x + MLP(LayerNorm(x))

    Two ideas make this work:

    LayerNorm ("pre-norm"): we normalize it before feeding it into each sublayer.
    It keeps activatiobs on a stable scale thru all 6 layers, so gradients flow well/training is stable.

    Residual connections the (" + x"): each sublayers output is added to its input, so information and gradients can bypass the sublayer entirely. This makes deep stacks easy to optimize -- the model starts near an identity map and learns small corrections on top.

    The order matters: this block does attention first (gather context) then the MLP (transforms that gathered context). Both add to x, it does not overwrite it.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int): 
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def __call__(self, x: Tensor, key_mask: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x), key_mask)
        x = x + self.mlp(self.ln2(x))
        return x 

