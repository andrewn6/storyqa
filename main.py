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
    person_loc: dict 
    item_loc: dict 
    holder: dict 
    obj_container: dict 

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
        inside [o for o in OBJECTS if self.obj_container.get(o) == container)]
        return inside or None 

    def loc(self, x: str) -> | None:
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
