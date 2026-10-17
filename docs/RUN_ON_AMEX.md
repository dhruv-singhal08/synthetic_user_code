# Run on the Amex laptop (SafeChain + Web UI)

Copy-paste PowerShell, run from the project folder (e.g.
`C:\Users\<you>\OneDrive - American Express\Desktop\synthetic-user`). Keep the folder name short
(Windows path limit). If something fails, send the **command + the full error text**.

## 1. Check Python and Poetry (one time)

```powershell
python --version          # must be 3.11.x or 3.12.x
py -0p                    # lists installed Pythons and their paths
poetry --version          # if not found: python -m poetry --version
```

If `python` is 3.13+, point Poetry at a 3.11/3.12 interpreter from the `py -0p` list:

```powershell
poetry env use "C:\path\to\Python312\python.exe"
```

## 2. Install (first time, and after `pyproject.toml` changes)

```powershell
poetry lock                    # resolves everything from Artifactory, writes poetry.lock
poetry install                 # add `--with gemini` only if you also want Gemini here
poetry show safechain
poetry show langchain-core
```

Expected: `safechain` version `1.5.0`, `langchain-core` version `0.3.81`.

## 3. Configure (one time)

```powershell
copy .env.example .env
copy "<path to your starter>\config\config_e2.yaml" config\safechain.yaml
notepad .env
```

Set these in `.env` (save the file!):

```
LLM_PROVIDER=                              # leave EMPTY: --config config/app.safechain.yaml picks SafeChain
SAFECHAIN_MODEL_ID=1                       # 1 = gpt-5.2, 3 = gpt-5-mini (keys in safechain.yaml)
SAFECHAIN_CONFIG_PATH=./config/safechain.yaml
CIBIS_CONSUMER_INTEGRATION_ID=<same value as in your starter .env>
CIBIS_CONSUMER_SECRET=<same value as in your starter .env>
CONFIG_PATH=./config/app.yaml              # this app's config - NOT the SafeChain yaml
```

**Your reference index:** copy your full index JSON (same format as
`data/references/reference_index_sample_v2.json`) to `data\references\reference_index.json` -
`config\app.safechain.yaml` already points there (`inputs.reference_index_file`). Remove that
line to run ungrounded (no index, no cheat sheets).

Check it loads before running anything else:

```powershell
poetry run synthetic-user validate-index --reference-index data/references/reference_index.json
```

Expected: `index_id=<your id> ... tables=11 columns=... relationships=... warnings=N` (no LLM
call), then one `WARNING` line per problem found. A load error names the exact file and field
path - send it to me.

Leave `GOOGLE_API_KEY` / `GEMINI_MODEL` empty unless you use Gemini. If you set
`LLM_PROVIDER=safechain`, it applies to *every* config, including the offline `app.fake.yaml` demo.

## 4. Verify (in this order - stop at the first failure)

```powershell
poetry run pytest -q                                              # offline tests: expect "... passed", 0 failed
poetry run synthetic-user smoke-llm --config config/app.safechain.yaml   # expect status=ok
poetry run synthetic-user run --config config/app.fake.yaml       # full offline pipeline, no LLM
```

## 5. Start the Web UI (real time, SafeChain)

```powershell
poetry run synthetic-user serve --config config/app.safechain.yaml
```

Open **http://127.0.0.1:8000** in Chrome/Edge. Leave the PowerShell window open - it shows the
live log of every step and LLM call. Stop the server with **Ctrl+C**.

- Port busy: `poetry run synthetic-user serve --config config/app.safechain.yaml --port 8001`
- More detail in the log: add `--log-level DEBUG`
- Offline demo (no SafeChain): `poetry run synthetic-user serve --config config/app.fake.yaml`

In the UI: create a run (optionally upload persona / product YAML and your reference index JSON),
then run the steps in order: **index -> scenarios -> questions -> execute -> evaluate**. Download the results CSV at
the end. Run folders are under `artifacts\runs\<run_id>\`.

## 6. CLI instead of the UI (optional)

```powershell
poetry run synthetic-user run --config config/app.safechain.yaml          # all steps
poetry run synthetic-user run --config config/app.safechain.yaml --from-step questions
poetry run synthetic-user --help
```

## What to send back if it fails

1. The command you ran.
2. The full error text from PowerShell (the last ~30 lines).
3. Output of `poetry run python --version`, `poetry show safechain`, `poetry show langchain-core`.

Never paste the `.env` file or the CIBIS secret.
