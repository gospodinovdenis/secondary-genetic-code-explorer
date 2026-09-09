# Secondary Genetic Code Explorer — real dissertation dataset

A Streamlit research prototype trained on the supplied `allalafullseq.xlsx` dataset.

## Dataset audit
- 18,063 original rows
- 20 amino-acid classes
- 6 species
- positions 1–85
- 6,727 sequence/class rows after exact deduplication
- 6,657 unique sequences
- 64 exact sequences occur with more than one amino-acid label, so evaluation is grouped by sequence

## Baseline validation
A 20% `GroupShuffleSplit`, grouped by exact sequence, produced:
- Accuracy: 0.8996
- Macro-F1: 0.8893

This prevents identical sequences from appearing in both training and test sets. It is a defensible baseline, not proof of cross-species biological generalization.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Google Colab
Open `Secondary_Genetic_Code_Explorer_Colab.ipynb`. Upload the original Excel file when prompted. The notebook parses it, audits duplicates, trains/evaluates the model, writes the Streamlit app, and shows commands for launching it through LocalTunnel.

## Interview framing
Describe this as a scientist-facing research prototype that converts a Ph.D. sequence-modeling workflow into standardized preprocessing, a validated baseline model, downloadable outputs, and a reusable interface. Do not call it a clinical or fully validated production model.
