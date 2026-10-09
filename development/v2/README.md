# Reviewed runtime source proposals

This directory preserves the selected source proposals as reviewable text. The
active core implementation and regression tests are in `src`; see the
[runtime guide](../../docs/v2/README.md) for APIs, defaults, and limitations.
Reference files are not imported or included in the installed package.

`source/inventory.json` lists all 104 selected references, content hashes, and final
integration decisions. Core features were adapted to the current runtime rather
than merged wholesale. The IkaGeneral experiment remains isolated and outside the
core scope. Excluded behavior is recorded explicitly.

The captured references were checked with Gitleaks v8.30.1 default rules and Python
references parse successfully. Zero scanner candidates cannot certify the absence
of every secret or determine whether code is confidential. Credential/configuration
files, logs, databases, unrelated runners, and exploit harnesses were excluded.

The superseded source branches were deleted without archive tags after the maintainer
accepted this inventory. Their deletion does not purge GitHub PR refs, old objects,
forks, caches, or other clones. Legacy history cleanup is a separate operation.
