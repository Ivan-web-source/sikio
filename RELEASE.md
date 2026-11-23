# Release checklist

Needs a PyPI API token (and one for TestPyPI). Never commit it; pass it with `UV_PUBLISH_TOKEN`.

1. Replace `OWNER` in `README.md`, `install.ps1` and `uninstall.ps1` with the GitHub user or organisation.
2. Search by hand for trademarks on "Sikio" (USPTO, EUIPO).
3. Bump `version` in `pyproject.toml`.
4. Run the tests: `python -m pytest -q test_sikio.py`
5. Build: `uv build`
6. Check: `uvx twine check dist/*`
7. Upload to TestPyPI first: `uv publish --publish-url https://test.pypi.org/legacy/`
8. Try it: `uv tool install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ sikio`
9. Upload for real: `uv publish`
10. Create the GitHub repo, push, tag `vX.Y.Z`, and make a GitHub release with the wheel attached.
11. Test the one-liner from the README on a clean Windows user account, then the uninstall line.
