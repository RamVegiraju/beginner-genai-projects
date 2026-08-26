# Setup

One-time setup for the whole series. Budget about five minutes.

## What you need

- A Databricks workspace with **Foundation Model APIs (pay-per-token)** enabled
- **Python 3.12**
- The **Databricks CLI**

## 1. Install the Databricks CLI

**macOS / Linux (Homebrew)**

```bash
brew tap databricks/tap
brew install databricks
```

**macOS / Linux (script)**

```bash
curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh
```

**Windows**

```powershell
winget install Databricks.DatabricksCLI
```

Check it worked:

```bash
databricks version
```

## 2. Log in

```bash
databricks auth login \
  --host https://<your-workspace>.cloud.databricks.com \
  --profile genai-series
```

Your browser opens, you approve, and the CLI saves a **profile** — a named
login — to `~/.databrickscfg`.

Two things that matter here:

- **Pass `--profile`.** It's required for the login to be saved properly.
- **Give it a descriptive name**, like `genai-series` or your workspace name.
  Avoid `DEFAULT`. A named profile makes it obvious which workspace a command
  is about to hit, and that's what stops you running something against the
  wrong one later.

Then point the samples at it:

```bash
export DATABRICKS_PROFILE=genai-series
```

Your workspace URL is the part of the address bar before the first `/` when
you're logged into Databricks.

## 3. Verify before writing any code

Two commands worth knowing — they answer "am I logged in?" and "as who?"

```bash
databricks auth profiles
```

```
Name           Host                                            Valid
genai-series   https://your-workspace.cloud.databricks.com     YES
```

**`Valid: YES` is what you're looking for.** `NO` means that profile's session
has expired — log in again.

```bash
databricks auth describe --profile genai-series
```

```
Host: https://your-workspace.cloud.databricks.com
User: you@example.com
Authenticated with: databricks-cli
Token storage: secure, OS keyring (service: databricks-cli)
```

Then confirm your workspace actually serves models:

```bash
databricks serving-endpoints list --profile genai-series
```

If that list is empty, Foundation Model APIs aren't enabled and nothing in
this series will run — sort that out first.

The samples default to `databricks-claude-haiku-4-5`. If it is unavailable—or
you prefer another model—choose an endpoint whose task is chat and whose state
is `READY`, then export its exact name:

```bash
export SERVING_ENDPOINT=<a-ready-chat-endpoint>
```

**Always list rather than trusting a model name you read somewhere.** Model
availability differs by workspace, new models land regularly, and old ones
retire. The list above is the source of truth for your workspace.

## 4. Python environment

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r 01-first-llm-call/requirements.txt
```

No `uv`? `pip install uv`, or use plain `python3.12 -m venv .venv` and
`.venv/bin/pip install -r ...`.

## 5. Run the first sample

```bash
export DATABRICKS_PROFILE=genai-series   # again if this is a new terminal
cd 01-first-llm-call
../.venv/bin/python invoke.py
```

That is the loop for every sample in the series: install its
`requirements.txt` from the repo root, `cd` in, run it.

---

## Best practice: how auth should work

**Use OAuth (`databricks auth login`). Don't use a personal access token.**

That's the single most important habit in this doc, and it's why no sample in
this repo contains a secret.

When you run `databricks auth login`, three good things happen:

1. **Tokens are short-lived.** The CLI mints an access token valid for about
   an hour and refreshes it automatically. A leaked one expires on its own.
2. **Nothing is stored in plaintext.** Recent CLI versions keep the credential
   in your **OS keyring** (Keychain on macOS), not in a readable file.
   `databricks auth describe` prints where yours lives.
3. **Your code stays clean.** The samples ask the CLI for a token at runtime:

   ```python
   w = WorkspaceClient(profile=PROFILE)
   token = w.config.authenticate()["Authorization"].removeprefix("Bearer ")
   ```

   There is no key to paste, leak, rotate, or accidentally commit.

A **personal access token (PAT)** is the opposite: a long-lived string, often
valid for months, that grants your full access to anyone holding it. PATs are
the most common way Databricks credentials leak — pasted into a notebook,
committed to a repo, dropped in Slack. Prefer OAuth.

### The rules

| Do | Don't |
|---|---|
| `databricks auth login --profile <name>` | Paste a PAT into your code |
| Read credentials from the CLI profile | Commit `.databrickscfg` or `.env` |
| Use `--profile` on every command | Assume which workspace you're hitting |
| Name profiles descriptively | Rely on `DEFAULT` |
| `databricks auth describe` when confused | Guess |

### Anything unattended is different

`databricks auth login` is **user-to-machine (U2M)** OAuth: it opens a browser,
so it needs a human. Databricks recommends it for interactive development,
which is exactly what this series is.

For CI, jobs, scheduled evaluations, and deployed apps there is no browser.
Use a **service principal** with **machine-to-machine (M2M)** OAuth — a
non-human identity with scoped permissions:

```bash
export DATABRICKS_HOST=https://<your-workspace>.cloud.databricks.com
export DATABRICKS_CLIENT_ID=<service-principal-client-id>
export DATABRICKS_CLIENT_SECRET=<service-principal-secret>
```

The SDK picks these up with no code change — the samples keep working as
written. `DATABRICKS_PROFILE` is optional; when absent, the SDK discovers
these ambient credentials. Two reasons it matters beyond "there's no browser":

- **No subprocess per client.** Under U2M the SDK shells out to the CLI for a
  token every time it builds a client. M2M mints tokens in-process instead.
- **It survives concurrency.** Tools that build many clients at once (an
  evaluation harness, a busy server) make those CLI calls race each other over
  the OS keyring. M2M has no keyring to contend over.

Still never a PAT.

## Working with more than one workspace

Every profile is a named login. Create as many as you need:

```bash
databricks auth login --host https://other-workspace.cloud.databricks.com --profile work
```

Then target one explicitly:

```bash
databricks serving-endpoints list --profile work
```

The samples read the `DATABRICKS_PROFILE` environment variable, so:

```bash
export DATABRICKS_PROFILE=work
```

Being explicit about the profile is a genuinely good habit — it's how you
avoid running something against the wrong workspace.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `401 Unauthorized` | Session expired | `databricks auth login --host <your-host>` |
| `Valid: NO` in `auth profiles` | Same | Same |
| `RESOURCE_DOES_NOT_EXIST` | Endpoint name typo, or not in your workspace | `databricks serving-endpoints list` |
| Empty endpoint list | Foundation Model APIs not enabled | Ask your workspace admin |
| `does not support the temperature parameter` | Newest reasoning models reject it | Drop `temperature` from the call |
| `command not found: databricks` | CLI not on `PATH` | Reopen your terminal, or reinstall |
| Wrong workspace hit | Ambiguous profile | `databricks auth describe` to see which one is active |
| `Use --profile to specify which profile to use` | Several profiles share one host, and `DATABRICKS_PROFILE` is unset | `export DATABRICKS_PROFILE=genai-series` |
