# catpie Development Documentation


## Setting up the dev environment


First checkout repo:

```shell
git clone https://github.com/dfsp-spirit/catpie
cd catpie/
```

Make sure you have `uv`, then install deps and run tests to ensure everything is working:

```shell
uv sync --group dev        # install package + dev deps (pytest)
uv run pytest              # unit tests (no R needed)
uv run python scripts/validate.py            # parity vs committed catR references
uv run python scripts/validate.py reference/catr_reference_goodbank.json  # strict BM/ML/WL
uv run python examples/demo.py               # demo
```

### Building the documentation

The user-friendly documentation (built with MkDocs + mkdocstrings) lives in
[`docs/`](docs/) and is published to GitHub Pages by the
[`Docs` workflow](.github/workflows/docs.yml). Build it locally with:

```shell
uv sync --group docs
uv run mkdocs build --strict   # outputs to site/
uv run mkdocs serve            # live preview at http://localhost:8000
```


### Making a new release

* Log changes in CHANGES and bump version
