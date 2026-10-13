# Contributing

Open an issue to discuss a bug or proposed change. Include the command, relevant
Hydra overrides, software versions, and a minimal reproducer. Never attach private
data, credentials, or model checkpoints to an issue or pull request.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
python -m compileall -q DyneTrion src openfold datasets metrics
```

Keep changes focused and document configuration or input-format changes. Add a
regression test when behavior changes. CPU tests do not replace a GPU check when
changing diffusion sampling or training. State explicitly which checks ran and
whether they used real pretrained weights.

The main actor uses supervised history score matching and environment support.
Preserve this distinction in code and documentation. Retain upstream copyright
headers and update THIRD_PARTY_NOTICES.md when adding third-party code.
