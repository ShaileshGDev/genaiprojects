# 20260429
### remove local commit and create fresh commit

Use `git reset --soft HEAD~1` to remove the latest local commit while keeping the changes staged, then create a fresh commit with the message you want. [geeksforgeeks](https://www.geeksforgeeks.org/git/how-to-delete-local-commit-in-git/)

## Commands

```bash
git reset --soft HEAD~1
git commit -m "your new message"
```

If you want the changes to be unstaged instead, use `git reset HEAD~1` first, then `git add .` and commit again. [sentry](https://sentry.io/answers/undo-the-most-recent-local-git-commits/)

## Notes

- This is safe for local, unpushed commits. [freecodecamp](https://www.freecodecamp.org/news/git-remove-last-commit-how-to-undo-a-commit-in-git/)
- If the commit was already pushed, rewriting history can affect others and usually needs a force push. [reddit](https://www.reddit.com/r/git/comments/sum03x/is_there_a_way_to_remove_all_local_commits_while/)

Would you like the version for removing the last **N** commits instead of just one?

### how to check what files in commit
Use `git show --name-only <commit>` to list the files changed in a commit. If you also want to see how each file was affected, use `git show --name-status <commit>`. [sentry](https://sentry.io/answers/list-all-files-in-a-git-commit/)

## Common commands

```bash
git show --name-only <commit>
git show --name-status <commit>
```

## Other useful options

- `git diff-tree --no-commit-id --name-only -r <commit>` lists changed files too. [geeksforgeeks](https://www.geeksforgeeks.org/git/how-to-list-all-files-in-a-commit-in-git/)
- `git log -1 --stat <commit>` gives a quick summary with file counts and line changes. [git-scm](https://git-scm.com/book/en/v2/Git-Basics-Viewing-the-Commit-History)

For the latest commit, `HEAD` works as the commit reference:

```bash
git show --name-only HEAD
```

### from HEAD need to remove 2 files, how to do that
To remove 2 files from the latest commit (`HEAD`) and make a fresh commit, amend the commit after unstaging or removing those files from the index. [blog.openreplay](https://blog.openreplay.com/git-remove-file-from-commit/)

## If the files should stay on disk
```bash
git reset --soft HEAD~1
git reset path/to/file1 path/to/file2
git commit -m "new commit message"
```

## If the files should be removed from the repo too
```bash
git rm path/to/file1 path/to/file2
git commit --amend --no-edit
```

`git rm --cached` is the right choice if you want to keep the files locally but stop tracking them in the commit. If this commit was already pushed, you’ll need to force-push after rewriting history. [blog.gitguardian](https://blog.gitguardian.com/rewriting-git-history-cheatsheet/)

For the exact case of “keep all other changes, remove only 2 files from HEAD,” the first command sequence is usually the safest. [stackoverflow](https://stackoverflow.com/questions/12481639/remove-file-from-latest-commit)

### my files are removed from the disk, but are in the commit, i want to remove the same from commit too

If the files are already deleted from disk but still “exist” in the latest commit, you can remove them from the commit by amending it (since `HEAD` still references them in the index). [reddit](https://www.reddit.com/r/git/comments/kzdf2z/removing_file_from_the_latest_commit/)

### 1. If the files are only in the last commit
```bash
git rm --cached path/to/file1 path/to/file2
git commit --amend --no-edit
```

`git rm --cached` removes them from Git’s index (so they disappear from the commit) but keeps their state as “untracked” or absent; `--amend` edits the top commit instead of creating a new one. [stackoverflow](https://stackoverflow.com/questions/1143796/remove-a-file-from-a-git-repository-without-deleting-it-from-the-local-filesyste)

### 2. Alternative using reset (keep other changes)
If you prefer to reconstruct the commit:

```bash
git reset --soft HEAD~1
git reset path/to/file1 path/to/file2   # unstage them
git commit -m "new message"
```

This keeps all other changes in the commit but drops the two files. [blog.gitguardian](https://blog.gitguardian.com/rewriting-git-history-cheatsheet/)

### Important note
Even after this, the files may still exist in earlier commits in the history; if you want them *fully* gone from all history (e.g., for secrets), you need `git filter‑repo`‑style rewriting. [dev](https://dev.to/morinoko/useful-git-commands-for-removing-accidentally-pushed-or-committed-files-2ld)

If you tell me whether this commit has been pushed yet, I can tailor the exact commands for your case.


### git list of all active commits not yet pushed
To list all commits that are local (active) but not yet pushed to the remote, use `git log` with a range that excludes the upstream branch. [geeksforgeeks](https://www.geeksforgeeks.org/git/how-to-list-unpushed-git-commits-local-but-not-on-origin/)

### For the current branch only

```bash
git log origin/main..HEAD
```

or, if your remote branch is named something else (e.g., `origin/master`):

```bash
git log origin/<branch-name>..HEAD
```

This shows all commits in your local branch that are not present on the remote. [educative](https://www.educative.io/answers/how-to-view-unpushed-git-commits)

### Short‑ened one‑line view

```bash
git log --oneline origin/main..HEAD
```

or use the shorthand for the upstream:

```bash
git log --oneline @{u}..
```

This lists only the unpushed commits with hash and subject on one line per commit. [linuxhint](https://linuxhint.com/list-git-commits-not-pushed-to-origin-yet/)

### All branches (all unpushed commits)

```bash
git log --branches --not --remotes
```

This lists commits that exist in any local branch but not in any remote branch. [betterstack](https://betterstack.com/community/questions/how-to-list-unpushed-git-commits/)


You’re using the commit *message* `"2025 files"` as the argument, but `git show` expects a commit hash or ref, not the message text—that’s why it errors with `ambiguous argument`. [geeksforgeeks](https://www.geeksforgeeks.org/git/how-to-list-all-files-in-a-commit-in-git/)

### 1. Show files changed in that commit (by hash)

Use the full or short hash from your log:

```bash
git show --name-only 2af05c888f52b5c9e98d750661d05daccef300d6
```

or shorten it:

```bash
git show --name-only 2af05c8
```

This lists only the filenames changed in that specific commit. [sentry](https://sentry.io/answers/list-all-files-in-a-git-commit/)

### 2. If you want only the filenames, no diffs

```bash
git show --name-only --oneline 2af05c8
```

or alternatively:

```bash
git log --pretty=format: --name-only -n 1 2af05c8
```

This gives you just the file‑paths touched in that “2025 files” commit. [stackoverflow](https://stackoverflow.com/questions/424071/how-do-i-list-all-the-files-in-a-commit)


The problem is that Git’s *history* still references those big files in the earlier un‑pushed commit, so `git push` tries to upload them even though they’re gone from disk. You must remove them from the commit history, not just from the working tree. [geeksforgeeks](https://www.geeksforgeeks.org/git/how-to-remove-a-large-file-from-commit-history-in-git/)

### 1. If you’re the only one working on this branch (and it’s not yet pushed)

You can rewrite the last commit so that it never contained those two files:

```bash
# First, ensure the big files are gone from index and working tree
git rm --cached 2025/sans-p-model/sanskrit-ocr/get_predictions/model.ckpt-190169.data-00000-of-00001
git rm --cached 2025/sans-p-model/sanskrit-ocr/get_predictions/model.ckpt-190169.meta

# Now re‑create the latest commit without them
git commit --amend --no-edit
```

Then push:

```bash
git push origin dev
```

This rewrites the commit at `2af05c8` so it never includes the model files, and then pushes the cleaned commit. [reddit](https://www.reddit.com/r/git/comments/qlgcjc/unintentionally_added_large_file_to_repo_and_now/)

### 2. If you want to *fully* purge them from all history (even older commits)

If those files appeared in earlier commits (even already‑pushed ones), you need to scrub them from the history. For just one or two big files, use `git filter‑repo` (modern, recommended) or `git filter‑branch` / `BFG`. Example with `git filter‑repo`: [community.latenode](https://community.latenode.com/t/github-push-fails-due-to-oversized-file-that-was-previously-removed/29914)

```bash
# Install git-filter-repo (if not present)
# Debian/Ubuntu: git clone https://github.com/newren/git-filter-repo ~/.git-filter-repo

git filter-repo --invert-paths --path 2025/sans-p-model/sanskrit-ocr/get_predictions/model.ckpt-190169.data-00000-of-00001
git filter-repo --invert-paths --path 2025/sans-p-model/sanskrit-ocr/get_predictions/model.ckpt-190169.meta
```

Then force‑push:

```bash
git push --force-with-lease origin dev
```

⚠️ This rewrites history, so coordinate with others if they’ve cloned the repo. [geeksforgeeks](https://www.geeksforgeeks.org/git/how-to-remove-a-large-file-from-commit-history-in-git/)

### 3. Prevention

Add patterns like:

```text
2025/**/model.ckpt-*
```

to `.gitignore` so future checkpoints are not tracked by Git. [baeldung](https://www.baeldung.com/ops/git-remove-file-commit-history)

If you tell me whether the “2025 files” commit (`2af05c8`) has been pushed anywhere or is still purely local, I can give you the minimal, safest exact command set for your case.

The push error is still happening because the large files are *still present in the history* of the commits you’re about to push, even though your latest `--amend` deletes them. GitHub reads the full pack and sees the 90+ MB and 140+ MB files somewhere in those “waiting to be pushed” commits. [docs.github](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)

***

### 1. Confirm what’s actually about to be pushed

List commits and their files before pushing:

```bash
git log --oneline origin/dev..HEAD
git log --name-only origin/dev..HEAD
```

This will show you exactly which commits (and which files!) are ready to go to `origin/dev`. [geeksforgeeks](https://www.geeksforgeeks.org/git/how-to-list-unpushed-git-commits-local-but-not-on-origin/)

***

### 2. Two options: remove or migrate to Git LFS

#### Option A: Remove those model files from history (recommended)

If you don’t need `model.ckpt-190169.*` in Git at all, you must rewrite the history so that **no commit ever contains them**. Simplest safe way for a mostly‑local branch:

```bash
git branch backup-dev  # backup before rewriting

git filter-repo --invert-paths \
  --path 2025/sans-p-model/sanskrit-ocr/get_predictions/model.ckpt-190169.data-00000-of-00001 \
  --path 2025/sans-p-model/sanskrit-ocr/get_predictions/model.ckpt-190169.meta
```

This removes them from all commits in the history. Then:

```bash
git push --force-with-lease origin dev
```

If `git filter-repo` is not installed, install it first (see `newren/git-filter-repo`; it’s the modern, recommended tool). [geeksforgeeks](https://www.geeksforgeeks.org/git/git-lfs-large-file-storage/)

#### Option B: Move them to Git LFS instead

If you *do* want to keep them in Git but avoid size limits, put them under Git LFS: [docs.github](https://docs.github.com/repositories/working-with-files/managing-large-files/about-git-large-file-storage)

```bash
git lfs install
git lfs track "**/model.ckpt-190169.*"
git add 2025/sans-p-model/sanskrit-ocr/get_predictions/model.ckpt-190169.data-00000-of-00001
git add 2025/sans-p-model/sanskrit-ocr/get_predictions/model.ckpt-190169.meta
git commit --amend --no-edit

git push origin dev
```

Now Git stores only small pointer files; LFS stores the big binaries. [graphite](https://graphite.com/guides/how-to-use-git-large-file-storage-lfs)

***

### 3. Prevention for the future

Add patterns like:

```text
2025/**/model.ckpt-*
```

to `.gitignore` so model checkpoints never get committed accidentally. [rewind](https://rewind.com/blog/overcoming-github-storage-limits/)

***

Given that you’re seeing the 90+ MB and 140+ MB files even after `--amend`, I strongly recommend **Option A (remove them from history with `git filter-repo`)** unless you specifically need them in Git. If you tell me whether you want to keep those model files at all (locally vs in Git), I can give you the exact filter‑repo or LFS command without copy‑paste errors.

### I WANT TO DISCARD MY COMMITS

You can discard your local commits and reset the branch back to the state of the remote (throwing away everything you added locally). Since you appear to be on `dev` and `origin/dev` is the “clean” point, do this:

```bash
git reset --hard origin/dev
```

This will:
- move `HEAD` back to where `origin/dev` is,
- discard all your local commits (including `a0f94b5` and anything after),
- also discard any local changes in tracked files (uncommitted work too).

If you want to **keep your working‑tree files** but just remove the commits, use:

```bash
git reset --soft origin/dev
```

This keeps all your changes as staged/uncommitted, but the branch pointer is reset to `origin/dev`. [stackoverflow](https://stackoverflow.com/questions/5097456/throw-away-local-commits-in-git)

⚠️ These operations are destructive for commits; if you think you might want to recover them later, first create a safety branch:

```bash
git branch backup-dev
```

then run the `reset --hard origin/dev` above. [geeksforgeeks](https://www.geeksforgeeks.org/git/how-to-delete-local-commit-in-git/)

