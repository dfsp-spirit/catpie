# Installation

catpie is a pure-Python package with **no runtime dependencies**, so
installation is trivial. We recommend the modern
[`uv`](https://docs.astral.sh/uv/) tool.

## 1. Install uv (if you don't have it)

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

or with pip:

```shell
pip install uv
```

## 2. Get the code

```shell
git clone https://github.com/dfsp-spirit/catpie.git
cd catpie
```

## 3. Create the environment and install

```shell
uv sync --group dev      # installs the package + pytest (for the tests)
```

This creates a virtual environment in `.venv/` and installs `catpie` (plus the
test and documentation tools). You can now use it:

```shell
uv run python -c "import catpie; print(catpie.__version__)"
```

## Using catpie in your own project

Once published to PyPI:

```shell
uv add catpie
```

Or, from a local checkout of this repository:

```shell
uv add --editable /path/to/catpie
```

## Running the checks

To confirm everything works:

```shell
uv run pytest                  # unit tests (35 tests, no R needed)
uv run python scripts/validate.py   # parity check against committed catR output
```

!!! info "Do I need R?"
    **No.** catpie is pure Python and never calls R. R (with the `catR`
    package) is only needed if you want to *regenerate* the ground-truth
    reference files from scratch — see [Parity with catR](validation.md).
