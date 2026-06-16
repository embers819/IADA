# Real-Feature Example

`examples/real_data/` contains four anonymized HER2 examples for smoke testing:

```text
real_data/
  labels.csv
  wsi/case_000.pt
  wsi/case_001.pt
  wsi/case_002.pt
  wsi/case_003.pt
  mri/case_000.h5
  mri/case_001.h5
  mri/case_002.h5
  mri/case_003.h5
```

These are real derived feature/volume files with anonymized filenames. They are
included only to verify the data loader and training loop. Do not treat metrics
from this four-case subset as meaningful.

Run the smoke test:

```bash
python train_iada.py --config configs/example_real_4cases.yaml
```

The command trains for one epoch on the four examples and evaluates on the same
four examples. Outputs are written to `runs/example_real_4cases/`.

Before uploading this repository publicly, confirm that these real derived data
files are approved for release. If not, remove `examples/real_data/` and keep
only this README plus the config format.
