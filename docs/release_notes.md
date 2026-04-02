# Release Notes

## Upcoming Release

### New Features and Major Changes

* [{Description of the fix} PR #XX](https://github.com/SPSUnipi/pySMSpp/pull/XX)

### Minor Changes and Bug Fixes

* 

## v0.2.0

### Minor Changes and Bug Fixes

* Include license file in the package distribution.

## v0.1.0 (First Public Release)

### New Features and Major Changes

* Initial release of `pypsa2smspp`, a Python package to convert 
  [PyPSA](https://pypsa.org/) networks to the 
  [SMS++](https://gitlab.com/smspp/smspp-project) format.
* Support for conversion of PyPSA `Network` objects to SMS++ input files.
* Support for Unit Commitment (UC) and Capacity Expansion problems via SMS++ `UCBlock` solved by `ucblock_solver`.
* Support for Capacity Expansion problems via SMS++ `InvestmentBlock` solved by `InvestmentBlock_test`.
* Support for the following PyPSA components:
  * Generators
  * Storage Units
  * Loads
  * Lines and Links
  * Buses
* Support for inverse conversion from SMS++ input files back to PyPSA `Network` objects populated with solution.
* Automated testing via `pytest`.

## Release Process

* Checkout a new release branch `git checkout -b release-v0.x.x`.
* Finalise release notes at `docs/release_notes.md`.
* Update version number in `pyproject.toml`.
* Open, review, and merge pull request for branch `release-v0.x.x`.
  Make sure to close issues and PRs or the release milestone with it (e.g. closes #X).
  Run `pre-commit run --all-files` locally and fix any issues.
* Update and checkout your local `main` and tag a release with `git tag v0.x.x`, `git push`, `git push --tags`.
  Include release notes in the tag message using the GitHub UI.
