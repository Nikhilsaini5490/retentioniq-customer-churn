# Data directory

The repository does not include a real customer dataset. Generate a deterministic **synthetic** dataset for demos with:

```powershell
python scripts/generate_demo_data.py --rows 2400 --seed 42
```

The generated CSV is ignored by Git. For a real academic evaluation, use a properly licensed public churn dataset or institution-approved data, document its source and consent/privacy constraints, and do not describe the synthetic demo metrics as real-world performance.