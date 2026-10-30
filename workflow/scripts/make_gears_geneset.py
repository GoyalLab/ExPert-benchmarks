#!/usr/bin/env python3
"""Build GEARS' custom perturbation-graph gene set for the padding mode.

GEARS can only score a perturbation that is a NODE in its perturbation graph.
`PertData(default_pert_graph=True)` builds that graph from
`essential_all_data_pert_genes.pkl` (9,976 genes), which covers just **961 of the
10,245** zero-shot perturbations — GEARS would be -inf on 90.6% of the class space,
worse than the untrained neg_target floor. `gene2go_all.pkl` covers **9,709**, so
handing GEARS `gene_set_path = (train_pool | zs_classes) & gene2go` (18,595 genes)
raises coverage to 94.8%.

This is also why GEARS must be RETRAINED for this mode rather than re-scored:
`gears.py` sets `num_perts = len(pert_list)` and `model.py` builds
`nn.Embedding(num_perts, hidden_size)`, so widening the graph changes the model's
parameter shape.
"""
import argparse
import json
import logging
import pickle

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gears_geneset")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", required=True, help="the padding class space (classes.txt)")
    ap.add_argument("--meta", required=True, help="padding metadata.json (for the training pool size)")
    ap.add_argument("--gene2go", required=True)
    ap.add_argument("--train-perts", default=None,
                    help="optional newline file of training-pool perturbations to include")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    with open(a.classes) as fh:
        zs = [l.strip() for l in fh if l.strip()]
    with open(a.gene2go, "rb") as fh:
        g2g = pickle.load(fh)
    g2g_genes = set(g2g)

    want = set(zs)
    if a.train_perts:
        with open(a.train_perts) as fh:
            want |= {l.strip() for l in fh if l.strip()}
    else:
        # No explicit training-pool file: include every gene2go gene so the graph
        # covers the training perturbations too. GEARS intersects with gene2go
        # anyway, and a wider node set only costs embedding rows.
        with open(a.meta) as fh:
            meta = json.load(fh)
        log.info("no --train-perts given; widening to all gene2go genes "
                 "(training pool is %s perts)",
                 (meta.get("train_space") or {}).get("n_train_perts", "?"))
        want |= g2g_genes

    gene_set = sorted(want & g2g_genes)
    cov = len(set(zs) & g2g_genes)
    log.info("gene set: %d genes | zs coverage %d/%d (%.1f%%)",
             len(gene_set), cov, len(zs), 100.0 * cov / max(1, len(zs)))
    if cov == 0:
        raise SystemExit("the gene set covers NO zero-shot perturbation — check --gene2go")

    with open(a.out, "wb") as fh:
        pickle.dump(gene_set, fh, protocol=pickle.HIGHEST_PROTOCOL)
    with open(a.out + ".json", "w") as fh:
        json.dump({"n_genes": len(gene_set), "n_zs_classes": len(zs),
                   "zs_in_graph": cov, "zs_coverage": cov / max(1, len(zs)),
                   "gene2go": a.gene2go}, fh, indent=1)
    log.info("wrote %s", a.out)


if __name__ == "__main__":
    main()
