# Tony's Blog

Source for [nltuan.github.io](https://nltuan.github.io), built with Pelican.

## Setup

Clone the repository and initialize the Elegant theme submodule:

```bash
git clone --recurse-submodules git@github.com:NLTuan/NLTuan.github.io.git
cd NLTuan.github.io
```

If you have already cloned the repository, run:

```bash
git submodule update --init --recursive
```

Install the locked Python dependencies and create the local virtual environment:

```bash
uv sync
```

## Preview locally

Build the site, then serve the generated `output/` directory:

```bash
uv run make html
uv run python -m http.server 8000 --directory output
```

Open <http://localhost:8000>. The development build uses relative URLs, so links and embedded media behave as they will on the published site.

## Publish

Commit your source changes, then generate the production site and push it to the `gh-pages` branch:

```bash
uv run make github
```

GitHub Pages serves the `gh-pages` branch. The `content/images/` directory is copied to the published site, including embedded videos.
