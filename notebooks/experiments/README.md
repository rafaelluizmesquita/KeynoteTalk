# Reconstructed XI SBGEA OSSE notebooks

The three self-contained notebooks are executable reconstructions of the OSSE analyses requested for the XI SBGEA keynote:

- `Experiment_1_FabryPerot_Cariri.ipynb` — synthetic OI 630.0 nm Fabry–Pérot measurements and thermospheric wind/temperature retrieval.
- `Experiment_2_MeteorRadar_CampinaGrande.ipynb` — meteor-radar geometry, echo-count, and multi-site wind retrieval sensitivity.
- `Experiment_3_GNSS_EquatorialSpreadF.ipynb` — synthetic GNSS slant-TEC observations and regularized equatorial depletion/network-design retrieval.

## Recovery and provenance

A resumed migration pass deleted the original source notebooks. Their original SHA-256 values are preserved in `KEYNOTE/MIGRATION_MANIFEST.csv`; the original notebook bytes could not be recovered. These replacement notebooks are therefore documented as **reconstructions**, not byte-identical restores. They combine the project requirements with preserved OSSE payloads, CSV sensitivity tables, figures, and the detailed scientific specification. The first two reconstructions were present in the working tree and are now re-executed; the GNSS source was rebuilt and its products regenerated.

## Reproduce

From the repository environment, run each notebook in order with:

```bash
export SBGEA_KEYNOTE_DATA="$HOME/Documents/Science/Programs/LFS/KEYNOTE"
jupyter nbconvert --to notebook --execute --inplace notebooks/experiments/Experiment_1_FabryPerot_Cariri.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/experiments/Experiment_2_MeteorRadar_CampinaGrande.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/experiments/Experiment_3_GNSS_EquatorialSpreadF.ipynb
```

The notebooks use NumPy, SciPy, pandas, Matplotlib, and Pillow. PyIRI is optional; the GNSS notebook checks its documented `IRI_density_1day` interface at runtime and falls back to a labeled Chapman-like background when PyIRI is unavailable. It does not use proprietary data, credentials, or external input files.

Generated OSSE data are written to `KEYNOTE/outputs/data/osse/`; slide figures and the 3-column composite are written to `KEYNOTE/outputs/figures/keynote/`. The composite is 11.25 × 5.5 inches at 300 DPI (3375 × 1650 px).

## Scientific scope

The FPI model is an illustrative Airy/CCD model, not a verified Cariri instrument specification. The meteor-radar sites are hypothetical independent monostatic systems. The GNSS observation geometry uses synthetic static satellite positions and a reduced horizontal depletion state with prescribed vertical structure; its network-layout evaluation is separated from the held-out bubble realizations. Assumptions, uncertainty definitions, model error, and limitations are documented inside each notebook.
