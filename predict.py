from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from molprop.checkpoints import load_model_from_checkpoint
from molprop.featurization import smiles_to_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a trained molecular GNN checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--smiles", type=str, action="append", default=[])
    parser.add_argument("--input-csv", type=str, default=None)
    parser.add_argument("--smiles-col", type=str, default="smiles")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-csv", type=str, default=None)
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_smiles(args: argparse.Namespace) -> list[str]:
    smiles = list(args.smiles)
    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        if args.smiles_col not in df.columns:
            raise ValueError(f"Missing column '{args.smiles_col}' in {args.input_csv}")
        smiles.extend(df[args.smiles_col].astype(str).tolist())
    if not smiles:
        raise ValueError("Provide at least one --smiles value or --input-csv.")
    return smiles


def invert_label_mapping(mapping: dict) -> dict[int, str]:
    inverse = {}
    for raw_label, idx in mapping.items():
        inverse[int(idx)] = str(raw_label)
    return inverse


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    smiles_list = load_smiles(args)

    model, payload = load_model_from_checkpoint(args.checkpoint, device=device)
    label_map = payload.get("extra", {}).get("label_to_index", {})
    index_to_label = invert_label_mapping(label_map) if label_map else {}

    rows = [
        {
            "smiles": smi,
            "is_valid_smiles": False,
            "pred_index": np.nan,
            "pred_label": None,
            "probability_positive": np.nan,
        }
        for smi in smiles_list
    ]

    valid_positions = []
    valid_graphs = []
    for i, smi in enumerate(smiles_list):
        graph = smiles_to_data(smi, label=None)
        if graph is None:
            continue
        valid_positions.append(i)
        valid_graphs.append(graph)

    if valid_graphs:
        loader = DataLoader(valid_graphs, batch_size=args.batch_size, shuffle=False)
        all_preds = []
        all_probs = []
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            probs = torch.softmax(logits, dim=1)
            all_preds.extend(probs.argmax(dim=1).cpu().numpy().tolist())
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())

        for pos, pred_idx, prob in zip(valid_positions, all_preds, all_probs):
            rows[pos]["is_valid_smiles"] = True
            rows[pos]["pred_index"] = int(pred_idx)
            rows[pos]["pred_label"] = index_to_label.get(int(pred_idx), str(pred_idx))
            rows[pos]["probability_positive"] = float(prob)

    result_df = pd.DataFrame(rows)
    print(result_df.to_string(index=False))

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_path, index=False)
        print(f"\nSaved predictions to {output_path}")


if __name__ == "__main__":
    main()

