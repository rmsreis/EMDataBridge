CONTRIBUTING
============

Thanks for contributing to this project. A couple of quick guidelines to keep the repository clean and reproducible.

Do NOT check built artifacts into the repository
------------------------------------------------

- The following are build artifacts and must NOT be committed to git:
  - `dist/` (sdist/wheel)
  - `build/`
  - `*.egg-info/`
  - compiled Python files and caches (`__pycache__/`, `*.pyc`)

- These files are ignored by `.gitignore`. If you accidentally committed them, remove them from the index with:

```sh
git rm -r --cached dist build *.egg-info
git commit -m "Remove build artifacts from VCS"
```

How to build and release
------------------------

- Build locally (recommended inside a virtualenv):

```sh
python -m pip install --upgrade build
python -m build
```

- Upload artifacts to a GitHub release (preferred) or to a package index. Artifacts in `dist/` are for distribution only and should not be checked in.

Working with branches and PRs
----------------------------

- Keep feature work on a branch and open a pull request against `main` (or the repo's default branch).
- Include tests for new behavior and update docs.

If you have questions or need a release created, open an issue or ping the maintainers.

Thank you — keeping the repo clean makes contributions and releases much safer and easier.
