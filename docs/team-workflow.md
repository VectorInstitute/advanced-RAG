# Share a team's work and pick up repository updates

Keep a team's notebooks and scripts on a branch of a team fork, keep the team's documents out of
git, and merge fixes from this repository into that branch.

The workspace checkout's `origin` is this repository over HTTPS, so `git pull` on `main` brings in
fixes. A team pushes its own work to a second remote, its fork, on a branch of its own.

## Prerequisites

- A fork of `VectorInstitute/advanced-RAG` on GitHub, owned by one teammate, with the others added
  as collaborators.
- On a Coder workspace, GitHub linked under **Account** > **External authentication**, followed by a
  workspace restart. The restart adds the workspace's SSH key to your GitHub account, which is what
  lets `git push` reach the fork. A push over HTTPS fails with HTTP 403.

## Set up the team branch

Run these from the repository root, `~/advanced-RAG` on a workspace.

1. Add the fork as a remote named `fork`:

   ```bash
   $ git remote add fork git@github.com:<fork_owner>/advanced-RAG.git
   ```

   where `<fork_owner>` is the GitHub handle that owns the fork.

2. Create the team branch from `main`:

   ```bash
   $ git switch -c <team_branch>
   ```

   where `<team_branch>` is a name such as `team/policy-qa`.

3. Push it, so teammates can check it out:

   ```bash
   $ git push -u fork <team_branch>
   ```

A teammate on another workspace adds the same remote, then runs `git fetch fork` and
`git switch <team_branch>`.

## Keep the documents out of git

Put the team's corpus outside the repository, for example in `~/team/corpus`, and point the
notebooks at that path. Files under `~/advanced-RAG` that you never `git add` stay out of commits,
but a later `git add .` or `git add -A` picks them up. Record the corpus's source and licence in a
file next to it, so a teammate knows what may be shared.

The caches under `~/.cache/aieng-advanced-rag` hold model responses that quote the documents. They
are outside the repository and stay on the workspace disk across restarts.

## Commit your work

Name the files you add instead of adding everything:

```bash
$ git add implementations/regulatory_policy_qa/<your_notebook>.ipynb team/
```

```bash
$ git commit -m "feat: add the policy corpus question set"
```

```bash
$ git push
```

A notebook copied from notebook 04 or 99 imports `byoc` and `advanced_lab` from
`implementations/regulatory_policy_qa/`, so keep the copy in that folder. A notebook elsewhere needs
that folder on `sys.path`.

Executed notebooks carry their outputs into the commit. The outputs can quote your documents, so
clear them before you commit if the corpus may not be shared.

## Merge updates from this repository

Fixes land on this repository's `main`. Commit your work first, then merge them into the team
branch:

```bash
$ git fetch origin
```

```bash
$ git merge origin/main
```

If the merge changed `uv.lock`, install the new dependencies and restart the notebook kernel:

```bash
$ uv sync --frozen --all-packages --all-extras --dev
```

A workspace restart runs the same install through `scripts/setup.sh`.

A cached result records a digest of the file that produced it. If the merge changed that file,
the cell that reads the result adds a `code changed` line, and with `RUN_EXPENSIVE = True` it
rebuilds the result, which spends model calls again. Set the flag to `False` in your copy to keep
reading the result you have.

Push the merged branch with `git push`, and teammates run `git pull` on the team branch.

## Verify

- `git remote -v` lists `origin` (this repository) and `fork` (the team fork).
- `git status -sb` shows the team branch tracking `fork/<team_branch>`.
- `git log --oneline -1 origin/main` matches the newest commit on this repository's `main` after a
  fetch.

## Troubleshoot

| Symptom | Cause | Fix |
| --- | --- | --- |
| `Permission denied (publickey)` on push | The workspace key is not on your GitHub account. | Link GitHub under **Account** > **External authentication** and restart the workspace. |
| `HTTP 403` on push | The `fork` remote uses HTTPS. | Set it to the SSH URL with `git remote set-url fork git@github.com:<fork_owner>/advanced-RAG.git`. |
| `There is no tracking information for the current branch` from `git pull` | The team branch was created without `-u`. | Run `git push -u fork <team_branch>` once. |
| A merge conflict in a notebook you did not edit | The update changed a notebook's outputs. | Take the incoming version with `git checkout --theirs <path>`, then `git add <path>`. |
| `ModuleNotFoundError` after a merge | The update added a dependency. | Run the `uv sync` command above and restart the kernel. |
