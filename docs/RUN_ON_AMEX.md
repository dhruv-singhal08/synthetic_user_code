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
PRODUCT_API_TOKEN=                         # only if your product contract uses ${PRODUCT_API_TOKEN}
```

Do **not** add `CONFIG_PATH` to `.env`: SafeChain reads `CONFIG_PATH` as *its* YAML and may
reload `.env`, which then points it at this app's config ("No models defined in config"). Pick
this app's config with `--config` instead, as every command below does.

**Your reference (any format) + usage guide:** the reference can be any `.json`, `.txt` or `.md`
file - your NL2SQL index JSON as it is, or e.g. a one-page table description for Voice of
Customer. Write a short usage guide (`.md`) saying how to read it and what counts as core
(template: `data/references/sample_card_spend_guide.md`). For the **CLI**, copy them to
`data\references\reference_index.json` (or any name) and set `inputs.reference_file` /
`inputs.reference_guide_file` in `config\app.safechain.yaml` (remove both lines to run the CLI
ungrounded). The **Web UI** does not read them: you upload both in the UI (toggle on =
grounded, off = ungrounded).

Check it loads before running anything else:

```powershell
poetry run synthetic-user check-reference --config config/app.safechain.yaml --reference data/references/reference_index.json --reference-guide data/references/my_guide.md
```

Expected: `reference_id=<file>-<hash> format=json chars=... max_chars=200000 guide=my_guide.md`
(no LLM call). If the file is larger than `max_chars`, trim it or raise `reference.max_chars`
in the config if the model's context window allows.

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

In the UI (top to bottom):

1. **Inputs** - Persona: keep the sample or upload your YAML; optionally paste sample queries
   (one per line). Product description: upload your YAML (required, no sample). Reference:
   switch **on** and upload your reference file (JSON / TXT / MD) and, optionally, its usage
   guide for a grounded run, or leave it **off** for an ungrounded run. Then **Start run**.
2. **Product connection** - enter the product URL, upload a contract (YAML/JSON: request
   template + where the answer is in the reply; **Download a contract template** shows the
   format) and click **Connect**. Put tokens in `.env` and reference them as
   `${PRODUCT_API_TOKEN}` in the contract headers - never paste a token into the file.
   **Mock product** is for demos only (an LLM stand-in plays the described product). Without a
   connection the configured fixture product answers, and it only knows the offline demo
   questions: live questions come back as `error` (recorded only).
   **Your NL2SQL API (nlsql agent, `POST /generate/sql/?timeout=10` on port 8007):** upload
   `config\products\generate_sql.adapter.yaml` as the contract; leave the URL empty to use
   `http://127.0.0.1:8007/generate/sql/?timeout=10` from the file, or type another one. Start the
   product first. If the call fails with a proxy error, add `NO_PROXY=localhost,127.0.0.1` to
   `.env` and restart. The UI and the product must not share a port (UI default 8000).
3. **Test pipeline** - Generate scenarios -> (optional: click a scenario row to review or edit
   its **Reference Pack**) -> Generate questions -> Run on product -> Evaluate, or **Run all
   remaining steps**.
4. **Results** - filter, search, click a row for details, **Download results.csv**.

Run folders are under `artifacts\runs\<run_id>\`.

## 6. CLI instead of the UI (optional)

```powershell
poetry run synthetic-user run --config config/app.safechain.yaml          # all steps (uses inputs.reference_file)
poetry run synthetic-user run --config config/app.safechain.yaml --reference <path\to\file> --reference-guide <path\to\guide.md>
poetry run synthetic-user run --config config/app.safechain.yaml --from-step questions
poetry run synthetic-user --help
```

## What to send back if it fails

1. The command you ran.
2. The full error text from PowerShell (the last ~30 lines).
3. Output of `poetry run python --version`, `poetry show safechain`, `poetry show langchain-core`.

Never paste the `.env` file or the CIBIS secret.
