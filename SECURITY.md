# Security policy

Report a security vulnerability in this repository privately, so it can be fixed before anyone
else learns of it. Don't open a public issue, pull request, or discussion about it.

## Report a vulnerability

Use GitHub's private vulnerability reporting: on the repository's **Security** tab, select
**Report a vulnerability**, and fill in the advisory form. GitHub's guide to
[privately reporting a security vulnerability](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
describes the form.

If you can't use the form, email advanced-rag@vectorinstitute.ai with "Security" in the subject.

A useful report names the file or commit affected, the steps that reproduce the problem, and what
an attacker gains. The maintainers reply on the advisory or by email.

## Scope

The policy covers the default branch, `main`, and the release assets of this repository.

In scope:

- The `aieng-advanced-rag` package and the code under `implementations/` and `scripts/`.
- The workspace setup: `scripts/setup.sh` and the `Dockerfile`.
- The reference-artifact installer, `python -m aieng.advanced_rag.cache`, and the
  `reference-artifacts.json` manifests, for example a way to install a file whose SHA-256 does not
  match its manifest.
- The GitHub Actions workflows in `.github/workflows/`.
- A credential, token, or personal data committed to the repository or attached to a release.

Out of scope:

- A vulnerability in a third-party dependency that is already public upstream. Report it to that
  project. Dependabot proposes the version bumps here.
- The content of the third-party datasets and models the notebooks download.
- Vector's hosted services that bootcamp workspaces use, such as the LLM proxy and the Coder
  deployment. Email advanced-rag@vectorinstitute.ai and the team passes the report on.
