# Release Notes

## Upcoming Release

### New Features and Major Changes

* Add TSSB marginal cost for IntermittentUnitBlocks [PR #40](https://github.com/SPSUnipi/pypsa2smspp/pull/40)
* Automatically handling new uncertain parameters for TSSB [PR #44](https://github.com/SPSUnipi/pypsa2smspp/pull/44)

### Minor Changes and Bug Fixes

* Add test for TSSB [PR #40](https://github.com/SPSUnipi/pypsa2smspp/pull/40)
* Fix of efficiency in time-dependent links [PR #43](https://github.com/SPSUnipi/pypsa2smspp/pull/43)

## v0.3.0

### New Features and Major Changes

* Improve configuration of the functions and distinction between thermal and intermittent units [PR #23](https://github.com/SPSUnipi/pypsa2smspp/pull/23)
* Add support for stochastic networks [PR #21](https://github.com/SPSUnipi/pypsa2smspp/pull/21) and [aa523b2](aa523b27453afed6cc6ed38282d279c86e3031a1)
* Support create/optimize/retrieve of SMS++ network [PR #27](https://github.com/SPSUnipi/pypsa2smspp/pull/27)
* Add documentation [PR #28](https://github.com/SPSUnipi/pypsa2smspp/pull/28)
* Support LineName, NodeName and UC name [PR #33](https://github.com/SPSUnipi/pypsa2smspp/pull/33)
* Support for time-dependent links [7522065](https://github.com/SPSUnipi/pypsa2smspp/tree/752206593202961f5809f0dde06fba50fef3795b)
* Support sector-coupled interface [7ac667a](7ac667af4d35875441fe29cb71096c343ac4f350),  [6730aa](6730daa0571261c89649adedac02c3a5bcfa41fb), [1fb41f2](1fb41f20a4bf5237f6b5a70aaaa67002d83aef99)

### Minor Changes and Bug Fixes

* Add architecture to documentation and improve examples [PR #37](https://github.com/SPSUnipi/pypsa2smspp/pull/37) and [PR #38](https://github.com/SPSUnipi/pypsa2smspp/pull/38)
* Improves git ignore and CI [PR #34](https://github.com/SPSUnipi/pypsa2smspp/pull/34)
* Cleaned dimensions for multi-links in sector-coupled networks
* Introduce CI with conda package
* Support for networks with snapshot weightings != 1 [cd973f3](ce973f3dc480441c9154dd6a268740b8d9c1cc4e)

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
