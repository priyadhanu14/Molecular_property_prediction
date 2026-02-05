from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from sklearn.ensemble import RandomForestClassifier
from tqdm.auto import trange

from molprop.checkpoints import load_model_from_checkpoint, save_checkpoint
from molprop.data import (
    MoleculeGraphDataset,
    create_data_loaders,
    load_smiles_dataframe,
    load_tox21_dataset,
    make_split_indices,
)
from molprop.featurization import compute_descriptor_vector, ecfp4_fingerprint
from molprop.models import build_model, supported_models
from molprop.training import compute_classification_metrics, evaluate, set_seed, train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train molecular property GNN models with MLflow tracking.")
    parser.add_argument("--data-path", type=str, default="data/input.csv")
    parser.add_argument("--smiles-col", type=str, default="smiles")
    parser.add_argument("--label-col", type=str, default="response")
    parser.add_argument("--download-tox21", action="store_true")
    parser.add_argument("--tox21-key", type=str, default="nr-arsmiles")

    parser.add_argument(
        "--model",
        type=str,
        default="attentivefp",
        choices=supported_models(),
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-timesteps", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--scheduler", type=str, default="onecycle", choices=("onecycle", "none"))

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--experiment-name", type=str, default="molecular-property-prediction")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--mlflow-tracking-uri", type=str, default="file:./mlruns")
    parser.add_argument("--mlflow-log-model", action="store_true")

    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--log-every", type=int, default=10)

    parser.add_argument("--rf-estimators", type=int, default=400)
    parser.add_argument("--train-rf-baseline", dest="train_rf_baseline", action="store_true")
    parser.add_argument("--no-rf-baseline", dest="train_rf_baseline", action="store_false")
    parser.set_defaults(train_rf_baseline=True)

    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    finite_metrics = {k: float(v) for k, v in metrics.items() if np.isfinite(v)}
    if finite_metrics:
        mlflow.log_metrics(finite_metrics, step=step)


def run_rf_baseline(
    smiles: list[str],
    labels: np.ndarray,
    split_indices: dict[str, np.ndarray],
    output_dir: Path,
    run_id: str,
    seed: int,
    n_estimators: int,
) -> tuple[dict[str, float], Path]:
    features = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError("Encountered invalid SMILES while building RF baseline features.")
        fp = ecfp4_fingerprint(mol)
        desc = compute_descriptor_vector(mol)
        features.append(np.concatenate([fp, desc], axis=0))
    X = np.vstack(features)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X[split_indices["train"]], labels[split_indices["train"]])

    metrics: dict[str, float] = {}
    for split in ("valid", "test"):
        idx = split_indices[split]
        probs = model.predict_proba(X[idx])[:, 1]
        preds = (probs >= 0.5).astype(np.int64)
        split_metrics = compute_classification_metrics(labels[idx], preds, probs)
        metrics.update({f"rf_{split}_{k}": v for k, v in split_metrics.items()})

    rf_path = output_dir / f"{run_id}_rf.joblib"
    joblib.dump(model, rf_path)
    return metrics, rf_path


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)

    data_path = Path(args.data_path)
    if args.download_tox21 and not data_path.exists():
        print(f"Downloading Tox21 data to {data_path} ...")
        load_tox21_dataset(data_path, key=args.tox21_key)

    df = load_smiles_dataframe(data_path, smiles_col=args.smiles_col, label_col=args.label_col).copy()
    df[args.label_col] = df[args.label_col].astype(int)

    unique_labels = sorted(df[args.label_col].unique().tolist())
    if len(unique_labels) != 2:
        raise ValueError(
            f"Expected binary labels, got {len(unique_labels)} classes: {unique_labels}. "
            "Current pipeline is configured for binary classification."
        )
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    df["label_index"] = df[args.label_col].map(label_to_index).astype(int)

    dataset = MoleculeGraphDataset(
        smiles=df[args.smiles_col].tolist(),
        labels=df["label_index"].tolist(),
        show_progress=not args.no_progress,
    )
    if len(dataset) == 0:
        raise RuntimeError("No valid molecules available after SMILES parsing.")

    # Keep the same row order as the processed dataset (invalid rows are dropped).
    valid_df = df.iloc[dataset.source_indices].reset_index(drop=True)

    split_indices = make_split_indices(
        num_examples=len(dataset),
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )
    _, loaders = create_data_loaders(
        dataset=dataset,
        split_indices=split_indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    node_dim = dataset[0].num_node_features
    edge_dim = dataset[0].num_edge_features
    model_kwargs = {
        "node_dim": node_dim,
        "edge_dim": edge_dim,
        "num_classes": 2,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "num_timesteps": args.num_timesteps,
    }
    model = build_model(args.model, **model_kwargs).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            steps_per_epoch=max(len(loaders["train"]), 1),
            epochs=args.epochs,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name):
        run = mlflow.active_run()
        run_id = run.info.run_id if run is not None else "unknown"

        mlflow.log_params(
            {
                "model": args.model,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "num_timesteps": args.num_timesteps,
                "dropout": args.dropout,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "scheduler": args.scheduler,
                "seed": args.seed,
                "train_ratio": args.train_ratio,
                "valid_ratio": args.valid_ratio,
                "num_examples": len(dataset),
                "dropped_invalid_smiles": len(dataset.dropped_indices),
                "data_path": str(data_path),
                "label_map": json.dumps(label_to_index),
            }
        )

        split_path = output_dir / f"{run_id}_splits.json"
        split_payload = {split: indices.tolist() for split, indices in split_indices.items()}
        split_path.write_text(json.dumps(split_payload, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(split_path), artifact_path="splits")

        best_val_loss = float("inf")
        best_checkpoint = output_dir / f"{run_id}_{args.model}_best.pt"

        epoch_range = trange(1, args.epochs + 1, disable=args.no_progress, desc="Training")
        for epoch in epoch_range:
            train_loss = train_one_epoch(
                model=model,
                loader=loaders["train"],
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                scheduler=scheduler,
            )
            valid_loss, valid_metrics, _, _, _ = evaluate(
                model=model,
                loader=loaders["valid"],
                criterion=criterion,
                device=device,
            )

            metrics = {"train_loss": train_loss, "valid_loss": valid_loss}
            metrics.update({f"valid_{k}": v for k, v in valid_metrics.items()})
            log_metrics(metrics, step=epoch)

            checkpoint_score = valid_loss if np.isfinite(valid_loss) else train_loss
            if checkpoint_score < best_val_loss:
                best_val_loss = checkpoint_score
                save_checkpoint(
                    checkpoint_path=best_checkpoint,
                    model=model,
                    model_name=args.model,
                    model_kwargs=model_kwargs,
                    extra={
                        "smiles_col": args.smiles_col,
                        "label_col": args.label_col,
                        "label_to_index": label_to_index,
                    },
                )

            if args.log_every > 0 and epoch % args.log_every == 0:
                print(
                    f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
                    f"| valid_loss={valid_loss:.4f} | valid_auc={valid_metrics['roc_auc']:.4f}"
                )

        best_model, checkpoint_payload = load_model_from_checkpoint(best_checkpoint, device=device)

        valid_loss, valid_metrics, _, _, _ = evaluate(best_model, loaders["valid"], criterion, device)
        test_loss, test_metrics, y_true, y_pred, y_prob = evaluate(
            best_model, loaders["test"], criterion, device
        )

        final_metrics = {
            "best_valid_loss": best_val_loss,
            "final_valid_loss": valid_loss,
            "final_test_loss": test_loss,
        }
        final_metrics.update({f"final_valid_{k}": v for k, v in valid_metrics.items()})
        final_metrics.update({f"final_test_{k}": v for k, v in test_metrics.items()})
        log_metrics(final_metrics)

        test_rows = valid_df.iloc[split_indices["test"]].copy().reset_index(drop=True)
        pred_df = pd.DataFrame(
            {
                "smiles": test_rows[args.smiles_col].tolist(),
                "label_true": y_true.tolist(),
                "label_pred": y_pred.tolist(),
                "probability_positive": y_prob.tolist(),
            }
        )
        pred_path = output_dir / f"{run_id}_{args.model}_test_predictions.csv"
        pred_df.to_csv(pred_path, index=False)

        mlflow.log_artifact(str(best_checkpoint), artifact_path="checkpoints")
        mlflow.log_artifact(str(pred_path), artifact_path="predictions")
        if args.mlflow_log_model:
            mlflow.pytorch.log_model(best_model, artifact_path="gnn_model")

        if args.train_rf_baseline:
            rf_metrics, rf_path = run_rf_baseline(
                smiles=valid_df[args.smiles_col].tolist(),
                labels=valid_df["label_index"].to_numpy(dtype=np.int64),
                split_indices=split_indices,
                output_dir=output_dir,
                run_id=run_id,
                seed=args.seed,
                n_estimators=args.rf_estimators,
            )
            log_metrics(rf_metrics)
            mlflow.log_param("rf_estimators", args.rf_estimators)
            mlflow.log_artifact(str(rf_path), artifact_path="checkpoints")

        print(f"Done. MLflow run ID: {run_id}")
        print(f"Best checkpoint: {best_checkpoint}")
        print(f"Checkpoint metadata keys: {list(checkpoint_payload.get('extra', {}).keys())}")


if __name__ == "__main__":
    main()
