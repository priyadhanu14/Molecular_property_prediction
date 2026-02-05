from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.loader import DataLoader

from molprop.checkpoints import load_model_from_checkpoint
from molprop.featurization import smiles_to_data


def choose_device(device_choice: str) -> torch.device:
    if device_choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_choice)


@st.cache_resource
def get_model(checkpoint_path: str, device_choice: str):
    device = choose_device(device_choice)
    model, payload = load_model_from_checkpoint(checkpoint_path, device=device)
    return model, payload, device


def parse_smiles_from_text(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def run_predictions(model, device, smiles_list: list[str]) -> pd.DataFrame:
    rows = [{"smiles": smi, "valid": False, "pred_index": None, "prob_positive": None} for smi in smiles_list]
    valid_positions = []
    valid_graphs = []

    for i, smi in enumerate(smiles_list):
        graph = smiles_to_data(smi, label=None)
        if graph is None:
            continue
        valid_positions.append(i)
        valid_graphs.append(graph)

    if not valid_graphs:
        return pd.DataFrame(rows)

    loader = DataLoader(valid_graphs, batch_size=256, shuffle=False)
    with torch.no_grad():
        preds = []
        probs = []
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            softmax = torch.softmax(logits, dim=1)
            preds.extend(softmax.argmax(dim=1).cpu().numpy().tolist())
            probs.extend(softmax[:, 1].cpu().numpy().tolist())

    for i, pred, prob in zip(valid_positions, preds, probs):
        rows[i]["valid"] = True
        rows[i]["pred_index"] = int(pred)
        rows[i]["prob_positive"] = float(prob)

    return pd.DataFrame(rows)


st.set_page_config(page_title="Molecular Property Predictor", page_icon="🧪", layout="wide")
st.title("🧪 Molecular Property Prediction Demo")
st.caption("RDKit + PyTorch Geometric inference from a saved checkpoint.")

with st.sidebar:
    st.header("Model")
    checkpoint_path = st.text_input("Checkpoint path", value="models/latest.pt")
    device_choice = st.selectbox("Device", options=["auto", "cpu", "cuda"], index=0)

    model = None
    model_payload = None
    device = None
    if checkpoint_path:
        ckpt = Path(checkpoint_path)
        if ckpt.exists():
            try:
                model, model_payload, device = get_model(str(ckpt), device_choice)
                st.success("Model loaded")
                st.write("Model:", model_payload.get("model_name", "unknown"))
            except Exception as exc:  # pragma: no cover - UI guard
                st.error(f"Failed to load checkpoint: {exc}")
        else:
            st.warning("Checkpoint path does not exist yet.")

st.subheader("Input SMILES")
default_example = "CCO\nc1ccccc1\nCC(=O)Oc1ccccc1C(=O)O"
smiles_text = st.text_area("One SMILES per line", value=default_example, height=140)
uploaded_csv = st.file_uploader("Optional CSV upload", type=["csv"])
smiles_col = st.text_input("CSV smiles column", value="smiles")

if st.button("Predict", type="primary"):
    if model is None:
        st.error("Load a valid checkpoint first.")
    else:
        smiles = parse_smiles_from_text(smiles_text)
        if uploaded_csv is not None:
            upload_df = pd.read_csv(uploaded_csv)
            if smiles_col not in upload_df.columns:
                st.error(f"Column '{smiles_col}' not found in uploaded CSV.")
                st.stop()
            smiles.extend(upload_df[smiles_col].astype(str).tolist())

        if not smiles:
            st.error("Please provide at least one SMILES string.")
            st.stop()

        output_df = run_predictions(model, device, smiles)
        st.success(f"Predicted {len(output_df)} molecules")
        st.dataframe(output_df, use_container_width=True)

        valid_rows = output_df[output_df["valid"] == True]  # noqa: E712
        if not valid_rows.empty:
            first_smiles = valid_rows.iloc[0]["smiles"]
            mol = Chem.MolFromSmiles(first_smiles)
            if mol is not None:
                st.subheader("Example Molecule")
                st.image(Draw.MolToImage(mol, size=(350, 300)), caption=first_smiles)

