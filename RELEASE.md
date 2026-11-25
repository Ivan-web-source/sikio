# Release checklist

Needs a PyPI API token (and one for TestPyPI). Never commit it; pass it with `UV_PUBLISH_TOKEN`.

1. If the repo moves to another account, update `Ivan-web-source` in `README.md`, `install.ps1` and `uninstall.ps1`.
2. Search by hand for trademarks on "Sikio" (USPTO, EUIPO).
3. Bump `version` in `pyproject.toml` and the same `AppVersion` in `installer.iss`.
4. Run the tests: `python -m pytest -q test_sikio.py`
5. Build: `Remove-Item dist -Recurse -ErrorAction SilentlyContinue; uv build`
6. Check: `uvx twine check dist/*`
7. Upload to TestPyPI first: `uv publish --publish-url https://test.pypi.org/legacy/`
8. Try it: `uv tool install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ sikio`
9. Upload for real: `uv publish`
10. Build the installer (Inno Setup 6): `& "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe" installer.iss`. It writes `dist\Sikio-Setup.exe`. (`sikio.ico` was made from the tray icon: `sikio.tray_image('listening').resize((256,256)).save('sikio.ico', sizes=[(16,16),(32,32),(48,48),(256,256)])`.)
11. Create the GitHub repo, push, tag `vX.Y.Z`, and make a GitHub release with the wheel and `dist\Sikio-Setup.exe` attached.
12. Test the one-liner and the installer from the README on a clean Windows user account, then the uninstall line.

The installer is unsigned, so SmartScreen warns. Free signing for open-source projects is available from the SignPath Foundation (signpath.org); apply once the repo is public.
