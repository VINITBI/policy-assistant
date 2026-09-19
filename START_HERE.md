# Start here

Everything you need is in this folder, including a ready `.env`. You don't need to edit any file.

## Before you start (once)
- **Python 3.11** from python.org. During install, tick **"Add python.exe to PATH"**.
- **Ollama** from ollama.com. You already have it; keep it running.
- About **20 GB** of free disk space and **16 GB of RAM**.

## Windows
1. Unzip this folder, for example to `C:\projects\policy-assistant`.
2. Double-click **`setup.bat`**. This is needed only once and takes 15-30 minutes. When it asks "Load the 4 demo policies", press **Y**.
3. Double-click **`run.bat`**. Your browser opens at http://localhost:8501. Keep the black window open while you use the app, and close it to stop.

## macOS / Linux
1. `bash setup.sh` (once)
2. `bash run.sh`

## Log in
| Who | User ID | Password | What you see |
|---|---|---|---|
| Admin | `admin` | `Admin@123` | Upload policies, audit trail, login sessions, dashboard, evaluation, users |
| Employee | `employee` | `Employee@123` | Policy selector, chat box, previous conversations |

**To add your own policies:** log in as admin, go to **Upload policy**, choose the PDF, and click **Upload & index**. Wait until **Manage policies** shows it as `active`. You can delete the demo policies on the same page.

**Swagger (API testing):** open http://localhost:8000/docs, click **Authorize**, and enter `admin` / `Admin@123`.

## If something goes wrong
| You see | Do this |
|---|---|
| "Python was not found" | Install Python 3.11 and tick "Add to PATH", then run `setup.bat` again |
| "Cannot reach the Policy Assistant service" | Start the app with `run.bat` |
| "The local policy database is already open in another program" | Close the `run.bat` window before running any script |
| Answers are slow | In `.env`, set `MULTI_QUERY_ENABLED=false` and `RERANKER_MODEL=BAAI/bge-reranker-base`, then restart `run.bat` |
| Anything else | Every error shows a reference ID. Search for it in `logs\app.log` |

For the full technical details, see `README.md`.
