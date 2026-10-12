# Installation

`difftree` requires Python 3.11 or newer. It depends on `jax`, `optax`,
`scikit-learn`, `scipy`, and `numpy`.

Install the latest version from GitHub with pip:

```bash
pip install git+https://github.com/jkitchin/jax-difftree.git
```

or add it to a [uv](https://docs.astral.sh/uv/) project:

```bash
uv add git+https://github.com/jkitchin/jax-difftree.git
```

JAX installs its CPU build by default. For GPU support, follow the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

## Development install

```bash
git clone https://github.com/jkitchin/jax-difftree.git
cd jax-difftree
uv sync                  # package plus dev tools (pytest, jupyter, matplotlib)
uv run pytest            # run the test suite
```

## Building the documentation

```bash
uv sync --group docs
uv run sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser. The example notebooks are
rendered with their stored outputs and are not re-executed during the build.
