# Release Notes

## Upcoming Release

### New Features and Major Changes

* 

### Minor Changes and Bug Fixes

* 

## Release Process

* Checkout a new release branch `git checkout -b release-v0.x.x`.
* Finalise release notes at `docs/release_notes.md`.
* Update version number in `pyproject.toml`.
* Open, review, and merge pull request for branch `release-v0.x.x`.
  Make sure to close issues and PRs or the release milestone with it (e.g. closes #X).
  Run `pre-commit run --all-files` locally and fix any issues.
* Update and checkout your local `main` and tag a release with `git tag v0.x.x`, `git push`, `git push --tags`.
  Include release notes in the tag message using the GitHub UI.
