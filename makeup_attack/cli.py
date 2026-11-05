"""Command line entry point: ``python -m makeup_attack <command>``.

    index     embed a reference gallery once and save it to .npz
    match     find the closest same-gender identity for a face
    run       match -> pick attack budget -> launch DiffAM on the input face
    estimate  print the similarity -> budget table
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .budget import estimate_budget, format_table
from .diffam_runner import PRESETS, DiffAMJob


def _add_budget_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--min-steps", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=40)


def _match(args):
    from .face_index import FaceIndex, embed_query, load_analyzer

    index = FaceIndex.load(args.index)
    app = load_analyzer(det_size=args.det_size)
    embedding, gender = embed_query(app, args.image)
    matches = index.query(
        embedding,
        gender=None if args.any_gender else gender,
        top_k=args.top_k,
        exclude=args.image,
    )
    return gender, matches


def cmd_index(args) -> int:
    from .face_index import FaceIndex, load_analyzer

    app = load_analyzer(det_size=args.det_size)
    index = FaceIndex.build(app, args.gallery, crop_fallback=not args.no_crop_fallback)
    index.save(args.out)
    counts = {g: int((index.genders == g).sum()) for g in sorted(set(index.genders))}
    print(f"saved {len(index)} embeddings to {args.out} {counts}")
    return 0


def cmd_match(args) -> int:
    gender, matches = _match(args)
    if not matches:
        print(f"no gallery entries for gender {gender}")
        return 1
    print(f"query gender: {gender}")
    for rank, m in enumerate(matches, 1):
        b = estimate_budget(m.similarity, args.min_steps, args.max_steps)
        print(f"{rank:>2}. {m.similarity:.4f}  [{b.tier}]  steps={b.steps}  "
              f"p(fool)={b.fooling_probability:.1%}  {m.path}")
    return 0


def cmd_run(args) -> int:
    args.top_k = 1
    gender, matches = _match(args)
    if not matches:
        print(f"no gallery entries for gender {gender}")
        return 1
    best = matches[0]
    budget = estimate_budget(best.similarity, args.min_steps, args.max_steps)
    print(f"nearest identity: {best.path} (similarity {best.similarity:.4f}, {budget.tier})")
    print(f"attack budget: {budget.steps} DDIM steps, predicted p(fool) {budget.fooling_probability:.1%}")
    job = DiffAMJob(
        diffam_dir=Path(args.diffam_dir),
        image=Path(args.image),
        checkpoint=Path(args.checkpoint),
        mode=args.mode,
        steps=budget.steps,
        exp_dir=args.exp,
    )
    return job.run(dry_run=args.dry_run)


def cmd_estimate(args) -> int:
    print(format_table(args.similarity, min_steps=args.min_steps, max_steps=args.max_steps))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="makeup_attack", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("index", help="embed a reference gallery")
    p.add_argument("--gallery", required=True, help="folder with male/ and female/ subfolders")
    p.add_argument("--out", default="cache/gallery.npz")
    p.add_argument("--det-size", type=int, default=640)
    p.add_argument("--no-crop-fallback", action="store_true",
                   help="skip images where detection fails instead of embedding them as crops")
    p.set_defaults(func=cmd_index)

    for name, func, helptext in (("match", cmd_match, "find nearest identities"),
                                 ("run", cmd_run, "match and launch DiffAM")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--image", required=True)
        p.add_argument("--index", default="cache/gallery.npz")
        p.add_argument("--det-size", type=int, default=640)
        p.add_argument("--any-gender", action="store_true", help="search the whole gallery")
        _add_budget_args(p)
        if name == "match":
            p.add_argument("--top-k", type=int, default=5)
        else:
            p.add_argument("--diffam-dir", default="third_party/DiffAM")
            p.add_argument("--checkpoint", required=True, help="fine-tuned DiffAM .pth")
            p.add_argument("--mode", choices=sorted(PRESETS), default="MT",
                           help="MT: adversarial makeup transfer, MR: makeup removal")
            p.add_argument("--exp", default="./runs/similarity_guided")
            p.add_argument("--dry-run", action="store_true", help="print the DiffAM command only")
        p.set_defaults(func=func)

    p = sub.add_parser("estimate", help="print the similarity -> budget table")
    p.add_argument("similarity", type=float, nargs="*", default=[0.95, 0.85, 0.75, 0.60, 0.45, 0.30])
    _add_budget_args(p)
    p.set_defaults(func=cmd_estimate)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
