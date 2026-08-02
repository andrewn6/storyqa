from __future__ import annotations 
from configparser import ConverterMapping
import os, re, math, json, argparse 
from dataclasses import dataclass

import numpy as np 
from tinygrad import Tensor, dtypes
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
                if world.loc(c) == pl and world.holds(p) is None and world.contents(c):
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
        _, p, q, it = act
        world.holder[it] = q
        return [p, "gave", "the", it, "to", q, "."]
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
    seg = [0] * (len(story_tokens) + 2) + [1] * (len(question_tokens) + 1)
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
        self.attn = Attention(d_model, n_heads)
        self.ln2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def __call__(self, x: Tensor, key_mask: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x), key_mask)
        x = x + self.mlp(self.ln2(x))
        return x 

class Encoder:
    """
    A stack of N transformer blocks plus a final LayerNorm.

    Why a stack? A single block can attend to tokens and mix them once, but reasoning about a story is iterative: layer 1 finds "who is where", layer 2 composes that into relations, deeper layers chain facts together (e.g object -> container -> locations, deeper layers chain facts together). Each block refines the represnation builtb y the one below it.

    Why a final LayerNorm? Because every block is pre-norm, the last block's output is un-normalized (its internal LN feeds the sublayers, not the output). We add one LN at the end so the represenation handed to the classification head has a stable, normalized scale.
    """
    def __init__(self, d_model: int, n_layers: int, n_heads: int, d_ff: int):
        self.blocks = [Block(d_model, n_heads, d_ff) for _ in range(n_layers)]
        self.ln_f = LayerNorm(d_model)

    def __call__(self, x: Tensor, key_mask: Tensor) -> Tensor:
        for b in self.blocks:
            x = b(x, key_mask)
        return self.ln_f(x)

class StoryQA:
    """
    Full BERT-Style encoder for our model!

    The flow:

    1. Embedding: each input token id becomes a vector by *summing* three embeddings:
        token_emb(id) -- what the word is.
        pos_emb(position) -- where in the sequence it is.
        seg_emb(segment) -- is it part of the STORY (0) or the QUESTION (1)
    Learned (not sinusoidal) embeddings, all of size d_model.

    2. Encoder: 6 pre-norm transformer blocks (bidirectional attention, no causal mask). The [cls] token at the position 0 has no meaning of its own, but beacuse attention is bidirectional it can gather information from the entire story/question - so by the end its representation summarizes the whole input.

    3. CLS Head, we can the final hidden state of [cls] at position 0, and projecti t to one logit per candidate answer token(locations, people, objects, containers, nobody/nothing). The argmax is the predicted answer.
    """
    def __init__(self, vocab_size: int, n_classes: int):
        self.tok_emb = Embedding(vocab_size, D_MODEL)
        self.pos_emb = Embedding(CTX, D_MODEL)
        self.seg_emb = Embedding(2, D_MODEL)
        self.encoder = Encoder(D_MODEL, N_LAYERS, N_HEADS, D_FF)
        self.head = Linear(D_MODEL, n_classes)

    def __call__(self, ids: Tensor, segs: Tensor, pad_mask: Tensor) -> Tensor:
        B, T = ids.shape 
        pos = Tensor.arange(T)
        x = self.tok_emb(ids) + self.pos_emb(pos) + self.seg_emb(segs)
        key_mask = (pad_mask * -1e9).reshape(B, 1, 1, T)
        x = self.encoder(x, key_mask)
        cls = x[:, 0]
        return self.head(cls)

# training utils

def build_key_mask(ids_np: np.ndarray) -> np.ndarray:
    """
    returns a float mask the SAME shape as ids: 1.0 where the token is real, 0.0 where it is [pad]. StoryQA converts this into the additive -1e9 key mask
    """
    return (ids_np != PAD_IDX).astype(np.float32)

def lr_schedule(step: int, total: int, warmup: int, lr_max: float, lr_min: float) -> float:
    """
    learning rate as a function of step, linear warmup then cosine delay

    Warmup (first `warmup` steps): lr ramps linearly from ~0 up to lr_max. At the
    very start the model is random and gradients are noisy, so a small lr avoids
    destabilizing it. This matters a lot for transformers.

    After warmup: lr follows a cosine curve from lr_max down to lr_min over the
    remaining steps. The smooth decay lets the optimizer take big steps while it
    is still finding the basin, then progressively finer steps to settle into it.
    
    GLM generated notes ^^
    """
    if step < warmup:
        return lr_max * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, progress)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))

def clip_grad_norm(params: list, max_norm: float) -> float:
    """
    Global gradient clipping. After back prop (backward() function) we measure the total norm of ALL gradients combined, and if it exceeds max_norm we scale every gradient down by max_norm/norm. This prevents rare huge-gradient steps (which can blow up training) while leaving normal steps untouched. Returns the pre-clip norm.
    """
    total = 0.0 
    for p in params:
        if p.grad is not None:
            total += float(p.grad.cast(dtypes.float32).square().sum().numpy())
    norm = math.sqrt(total)
    if norm > max_norm:
        scale = max_norm / (norm + 1e-6)
        for p in params:
            if p.grad is not None:
                p.grad = p.grad * scale
    return norm

def count_params(model) -> int:
    return sum(int(p.numel()) for p in get_parameters(model))

def save_ckpt(model, step: int, best_acc: float):
    os.makedirs(CKPT_DIR, exist_ok=True)
    safe_save(get_state_dict(model), CKPT_PATH)
    with open(META_PATH, "w") as f:
        json.dump({"step": step, "best_acc": float(best_acc)}, f)

def load_ckpt(model) -> tuple[int, float]:
    if not (os.path.exists(CKPT_PATH) and os.path.exists(META_PATH)):
        return 0, 0.0 
    load_state_dict(model, safe_load(CKPT_PATH))
    with open(META_PATH) as f:
        meta = json.load(f)
    return int(meta["step"]), float(meta["best_acc"])

@dataclass 
class EvalResult:
    acc: float 
    loss: float

def evaluate(model, rng: np.random.Generator, n_batches: int, batch: int) -> EvalResult:
    """
    Average exact match accuracy and loss over batches of diff-2 (hardest)
    """
    acc_sum, loss_sum = 0.0, 0.0
    for _ in range(n_batches):
        ids, segs, labs, _ = make_batch(rng, batch, difficulty=2)
        out = model(Tensor(ids), Tensor(segs), Tensor(build_key_mask(ids)))
        loss = out.sparse_categorical_crossentropy(Tensor(labs))
        pred = out.argmax(axis=-1).numpy()
        acc_sum += float((pred == labs).mean())
        loss_sum += float(loss.numpy())
    return EvalResult(acc=acc_sum / n_batches, loss=loss_sum / n_batches)

def run_eval(model, rng: np.random.Generator, n_batches: int, batch: int) -> EvalResult:
    acc_sum, loss_sum = 0.0, 0.0 
    for _ in range(n_batches):
        ids, segs, labs, _ = make_batch(rng, batch, difficulty=2)
        out = model(Tensor(ids), Tensor(segs), Tensor(build_key_mask(ids)))
        loss = out.sparse_categorical_crossentropy(Tensor(labs))
        pred = out.argmax(axis=-1).numpy()
        acc_sum += float((pred == labs).mean())
        loss_sum += float(loss.numpy())
    return EvalResult(acc=acc_sum / n_batches, loss=loss_sum / n_batches)

def train(args):
    rng = np.random.default_rng(args.seed)
    Tensor.training = True
    model = StoryQA(VOCAB_SIZE, N_CLASSES)
    params = get_parameters(model)
    print(f"params: {count_params(model)/1e6:.2f}m | vocab {VOCAB_SIZE} | classes {N_CLASSES} | ctx {CTX}")


    
    start_step, best_acc = 0, 0.0 
    if args.resume and os.path.exists(CKPT_PATH):
          start_step, best_acc = load_ckpt(model)
          print(f"resumed from step {start_step} (best_acc {best_acc:.4f})")
    opt = AdamW(params, lr=args.lr, b1=0.9, b2=0.98, eps=1e-8, weight_decay=args.weight_decay)
    warmup = max(50, args.steps // 20)
    val_rng = np.random.default_rng(999)

    for step in range(start_step, args.steps):
        progress = step / max(1, args.steps)
        lr = lr_schedule(step, args.steps, warmup, args.lr, args.lr * 0.01)
        opt.lr.assign(Tensor([lr], device=opt.lr.device))

        ids, segs, labs, _ = make_batch(rng, args.batch, progress=progress)
        out = model(Tensor(ids), Tensor(segs), Tensor(build_key_mask(ids)))
        loss = out.sparse_categorical_crossentropy(Tensor(labs))

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % args.log_every == 0 or step == args.steps - 1:
            pred = out.argmax(axis=-1).numpy()
            acc = float((pred == labs).mean())
            print(f"step {step:5d} | lr {lr:.2e} | loss {float(loss.numpy()):.4f} | acc {acc:.3f}")

        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            res = run_eval(model, val_rng, n_batches=20, batch=args.batch)
            print(f"  -> val acc {res.acc:.4f} | val loss {res.loss:.4f}")
            if res.acc > best_acc:
                best_acc = res.acc
            save_ckpt(model, step + 1, best_acc)

    save_ckpt(model, args.steps, best_acc)
    print(f"done. best val acc {best_acc}:.4f")

def evaluate(args):
    rng = np.random.default_rng(2024)
    model = StoryQA(VOCAB_SIZE, N_CLASSES)
    if os.path.exists(CKPT_PATH):
        load_ckpt(model)
        print("loaded checkpoint")
    res = run_eval(model, rng, n_batches=args.n_batches, batch=args.batch) 
    print(f"val acc {res.acc:.4f} | val loss {res.loss:.4f} over {args.n_batches * args.batch} examples")

def _print_attn(model, seq: list[str]):
    real = [i for i in range(len(seq)) if seq[i] != "[pad]"]
    toks = [seq[i] for i in real]
    print("\n[cls] attention (top attended tokens per head):")
    for li, block in enumerate(model.encoder.blocks):
        attn = block.attn.last_attn
        if attn is None:
            continue
        w = attn[0, :, :].numpy()
        print(f"layer {li}")
        for h in range(w.shape[0]):
            row = [(toks[j], float(w[h, real[j]])) for j in range(len(real))]
            row.sort(key=lambda r: r[1], reverse=True)
            top = row[:6]
            print(f" head{h}: " + " ".join(f"{t}:{v:.2f}" for t, v in top))



def inspect(args):
    rng = np.random.default_rng(args.seed)
    model = StoryQA(VOCAB_SIZE, N_CLASSES)
    if os.path.exists(CKPT_PATH):
        load_ckpt(model)
        print("loaded checkpoint")
    ids, segs, labs, seqs = make_batch(rng, args.n, difficulty=args.difficulty)
    out = model(Tensor(ids), Tensor(segs), Tensor(build_key_mask(ids)))
    preds = out.argmax(axis=-1).numpy()

    for i in range(args.n):
        seq = seqs[i]
        sep1 = seq.index("[sep]")
        sep2 = len(seq) - 1 - seq[::-1].index("[sep]")
        story = seq[1:sep1]
        question = seq[sep1 + 1:sep2]
        expected = ANSWER_TOKENS[labs[i]]
        predicted = ANSWER_TOKENS[preds[i]]
        print("=" * 70)
        print(f"STORY:     {detokenize(story)}")
        print(f"QUESTION:  {detokenize(question)}")
        print(f"EXPECTED:  {expected}")
        print(f"PREDICTED: {predicted}  {'OK' if predicted == expected else 'WRONG'}")
        one_ids = np.array([ids[i]], dtype=np.int32)
        one_segs = np.array([segs[i]], dtype=np.int32)
        model(Tensor(one_ids), Tensor(one_segs), Tensor(build_key_mask(one_ids)))
        _print_attn(model, seq)

def main():
    p = argparse.ArgumentParser(description="story-qa transformer encoder (tinygrad)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train")
    pt.add_argument("--steps", type=int, default=2000)
    pt.add_argument("--batch", type=int, default=64)
    pt.add_argument("--lr", type=float, default=3e-4)
    pt.add_argument("--weight-decay", type=float, default=0.01)
    pt.add_argument("--clip", type=float, default=1.0)
    pt.add_argument("--seed", type=int, default=0)
    pt.add_argument("--log-every", type=int, default=50)
    pt.add_argument("--eval-every", type=int, default=200)
    pt.add_argument("--resume", action="store_true")
    pt.set_defaults(func=train)

    pe = sub.add_parser("evaluate")
    pe.add_argument("--batch", type=int, default=64)
    pe.add_argument("--n-batches", type=int, default=50)
    pe.set_defaults(func=evaluate)

    pi = sub.add_parser("inspect")
    pi.add_argument("--n", type=int, default=3)
    pi.add_argument("--seed", type=int, default=7)
    pi.add_argument("--difficulty", type=int, default=2, choices=[0, 1, 2])
    pi.set_defaults(func=inspect)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()

