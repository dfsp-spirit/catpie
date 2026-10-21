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

* Log changes in CHANGES and bump the version in `pyproject.toml`.
* Build the source and wheel distributions from the repo root:

```shell
uv build         # writes dist/catpie-<version>.tar.gz and dist/catpie-<version>-py3-none-any.whl
```

* Upload to TestPyPI first (`uv` needs no extra tooling, the token is created at
  https://test.pypi.org/manage/account/token/):

```shell
uv publish --publish-url https://test.pypi.org/legacy/ --token pypi-<your-testpypi-token>
```

* Verify the test upload by installing it in a throwaway environment:

```shell
uv run --isolated --no-project --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ catpie==<version>
```

* Upload to the real PyPI:

```shell
uv publish --token pypi-<your-pypi-token>
```

Notes:

* The token can also come from the `UV_PUBLISH_TOKEN` environment variable, which keeps it
  out of the shell history; `uv` uses `__token__` as the username automatically.
* A version number can only be uploaded to PyPI once. If the upload fails, bump the version
  instead of re-using it.
* The classic `twine` workflow works too, if preferred:
  `uv run --with twine twine upload --repository testpypi dist/*` (TestPyPI) and
  `uv run --with twine twine upload dist/*` (PyPI). This requires a `testpypi` entry in
  `~/.pypirc`.
* Create a GitHub release for the tag afterwards — the `Docs` workflow deploys the
  documentation on release.
