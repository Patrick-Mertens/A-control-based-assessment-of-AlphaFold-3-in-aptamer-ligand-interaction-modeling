# A control-based assessment of AlphaFold 3 in aptamer-ligand interaction modeling

This repository contains supporting information, processed data, and reproducibility scripts associated with the study **“A control-based assessment of AlphaFold 3 in aptamer-ligand interaction modeling.”**

## Repository structure

```text
.
├── Supporting_Information/
│   └── Supporting figures and tables associated with the manuscript
│
├── inputs/
│   └── aptamer_sequences.csv
│
├── scripts/
│   ├── generate_af3_inputs.py
│   ├── run_stage1_msa.sh
│   ├── run_stage2_inference.sh
│   ├── analyze_af3_outputs.py
│   ├── extract_publication_data.py
│   └── generate_supporting_tables.py
│
└── data/
    ├── per_aptamer_per_condition.csv
    ├── per_prediction_long.csv
    ├── per_prediction_values.csv
    └── run_metadata.json
```

## Data

The `data/` directory contains the processed AlphaFold 3 results used for downstream analysis.

The complete raw AlphaFold 3 output archive is approximately 8.3 GB and is therefore not stored directly in this GitHub repository.

## Reproducibility scripts

The scripts cover the main analysis workflow:

1. Generation of AlphaFold 3 JSON inputs.
2. Stage 1 MSA/data-pipeline execution.
3. Stage 2 AlphaFold 3 inference.
4. Extraction and analysis of AlphaFold 3 outputs.
5. Generation of publication-level processed datasets.
6. Generation of supporting tables.

Additional documentation describing the exact commands and computational environment used for the study will be added to this repository.

## Contact

For questions regarding the study or repository:

[patrick.ryan.mertens@caretronic.com](mailto:patrick.ryan.mertens@caretronic.com)

