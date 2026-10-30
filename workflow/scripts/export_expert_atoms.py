#!/usr/bin/env python3
"""Export a trained ExPert model's class-embedding table as a drop-in gene-embedding
feature set (docs/padding_scope.md §12, the "sharp test").

Question it answers: the simple forward baselines (GenePert, scouter) predict an
unseen perturbation's response from a *text* embedding of the target gene. ExPert
predicts it from a LEARNED class embedding built over the same graph. If we hand the
simple models ExPert's learned table instead of the text table and they improve,
then a large part of what padding buys is a better perturbation REPRESENTATION, and
any model can borrow it. If they do not improve, the gain is in ExPert's
cell-encoder / scoring path, not in the representation. Either outcome is a result.

The tensor is read straight from the checkpoint. Row order is NOT the npz order:
the vocabulary is
    labels.categorical_mapping (the trained classes, in registry order)
    ++ sorted(remaining genes in the model's class-embedding npz)
verified on the fold4 model by exact cosine match of `cls_emb._atom_text` rows
against that npz (40/40 random rows, cos = 1.0000).

Emits BOTH formats the consumers want:
    <out>.npz  {embeddings: (n, d) float32, genes: (n,) object}   -> GenePert --embeddings
    <out>.pkl  {gene: vec}                                        -> scouter build_genept_embd
"""
import argparse
import json
import logging
import os
import pickle

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("expert_atoms")


def build_vocab(trained, npz_genes):
    """trained classes (registry order) ++ sorted(everything else)."""
    padded = sorted(set(npz_genes) - set(trained))
    return list(trained) + padded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-pt", required=True, help=".../config/ExPert_v0/model/model.pt")
    ap.add_argument("--class-emb-npz", default=None,
                    help="the model's class-embedding npz (default: sibling gene_embeddings.npz)")
    ap.add_argument("--tensor", default="cls_emb.embedding.weight",
                    help="state_dict key holding the per-class embedding table")
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--out-pkl", default=None)
    ap.add_argument("--verify-rows", type=int, default=40,
                    help="random rows to verify the vocabulary order against the npz (0 = skip)")
    ap.add_argument("--l2-normalise", action="store_true",
                    help="row-normalise the exported table (ridge/MLP features are scale-sensitive)")
    a = ap.parse_args()

    import torch
    try:
        ck = torch.load(a.model_pt, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:                                   # torch < 2.1
        ck = torch.load(a.model_pt, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    if a.tensor not in sd:
        raise SystemExit(f"{a.tensor!r} not in state_dict. Candidates: "
                         f"{[k for k in sd if 'cls_emb' in k or 'class' in k]}")
    T = sd[a.tensor].float().numpy()

    trained = np.asarray(
        ck["attr_dict"]["registry_"]["field_registries"]["labels"]["state_registry"]
        ["categorical_mapping"]).tolist()

    npz_path = a.class_emb_npz or os.path.join(os.path.dirname(a.model_pt), "gene_embeddings.npz")
    d = np.load(npz_path, allow_pickle=True)
    npz_genes = np.asarray(d["genes"], dtype=object).tolist()

    vocab = build_vocab(trained, npz_genes)
    if len(vocab) != T.shape[0]:
        raise SystemExit(f"vocabulary is {len(vocab)} names but {a.tensor} has {T.shape[0]} rows "
                         f"(trained={len(trained)}, npz={len(npz_genes)})")
    log.info("%s: %s | vocab = %d trained ++ %d padded",
             a.tensor, T.shape, len(trained), len(vocab) - len(trained))

    # Verify the row order against the checkpoint's own text atoms.
    if a.verify_rows and "cls_emb._atom_text" in sd:
        at = sd["cls_emb._atom_text"].float().numpy()
        E = np.asarray(d["embeddings"], dtype=np.float32)
        if at.shape == E.shape:
            En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
            rng = np.random.default_rng(0)
            probe = rng.choice(len(vocab), min(a.verify_rows, len(vocab)), replace=False)
            An = at[probe] / (np.linalg.norm(at[probe], axis=1, keepdims=True) + 1e-12)
            S = An @ En.T
            bad = [(int(i), npz_genes[int(S[k].argmax())], vocab[i], float(S[k].max()))
                   for k, i in enumerate(probe)
                   if npz_genes[int(S[k].argmax())] != vocab[i] or S[k].max() < 0.999]
            if bad:
                raise SystemExit(f"vocabulary order does NOT match the checkpoint: {bad[:5]}")
            log.info("row order verified on %d random rows (all cos > 0.999)", len(probe))
        else:
            log.warning("_atom_text %s != npz %s — skipping order verification", at.shape, E.shape)

    if a.l2_normalise:
        T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12)

    os.makedirs(os.path.dirname(os.path.abspath(a.out_npz)), exist_ok=True)
    np.savez_compressed(a.out_npz,
                        embeddings=T.astype(np.float32),
                        genes=np.asarray(vocab, dtype=object))
    log.info("wrote %s", a.out_npz)

    if a.out_pkl:
        with open(a.out_pkl, "wb") as fh:
            pickle.dump({g: T[i].astype(np.float32) for i, g in enumerate(vocab)}, fh,
                        protocol=pickle.HIGHEST_PROTOCOL)
        log.info("wrote %s", a.out_pkl)

    meta = {"model_pt": a.model_pt, "tensor": a.tensor, "dim": int(T.shape[1]),
            "n_classes": int(T.shape[0]), "n_trained": len(trained),
            "n_padded": len(vocab) - len(trained), "l2_normalised": bool(a.l2_normalise),
            "class_emb_npz": npz_path}
    with open(os.path.splitext(a.out_npz)[0] + ".json", "w") as fh:
        json.dump(meta, fh, indent=1)


if __name__ == "__main__":
    main()
