# backend/main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import subprocess
import json
import sys
from typing import Optional
import time
import uuid
from threading import Thread

app = FastAPI(title="LegalSummarizer Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SCRIPTS_DIR = BASE_DIR / "backend" / "scripts"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Track pipelines by session_id
PIPELINE_PROGRESS = {}

@app.get("/")
def root():
    return {"message": "Backend is running! Use POST /run_pipeline"}

def run_script(script_path: Path, args: list = [], verbose=False, stage_name=None):
    cmd = [sys.executable, str(script_path)] + args
    if verbose and stage_name:
        print(f"▶ Stage: {stage_name} - Running {script_path.name} with args: {args}")
    elif verbose:
        print(f"▶ Running {script_path.name} with args: {args}")

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end = time.time()

    if verbose and stage_name:
        print(f"✅ Stage: {stage_name} completed in {end-start:.2f}s")
        if result.stdout:
            print(result.stdout)
    elif verbose:
        print(f"✅ Finished {script_path.name} in {end-start:.2f}s")
        if result.stdout:
            print(result.stdout)

    if result.returncode != 0:
        raise RuntimeError(
            f"Script failed: {' '.join(cmd)}\nExit code: {result.returncode}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )
    return result.stdout

def safe_load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

@app.post("/run_pipeline")
async def run_pipeline(dataset: str = Form(...), n: int = Form(...), entry_id: Optional[int] = Form(None), file: Optional[UploadFile] = File(None)):
    dataset = dataset.strip().upper()
    if dataset not in ["ILC", "IN-ABS"]:
        raise HTTPException(status_code=422, detail="Dataset must be 'ILC' or 'IN-ABS'")
    if n <= 0 or n > 1000:
        raise HTTPException(status_code=422, detail="n must be between 1 and 1000")

    session_id = str(uuid.uuid4())
    PIPELINE_PROGRESS[session_id] = {
        "stages": [], 
        "completed": False, 
        "results": None, 
        "error": None,
        "entry_id": entry_id  # Store entry_id for later filtering
    }

    uploaded_path = None
    if file is not None:
        if not file.filename.lower().endswith(".json"):
            raise HTTPException(status_code=422, detail="Uploaded file must be a .json")
        uploaded_path = DATA_DIR / "custom_input.json"
        contents = await file.read()
        uploaded_path.write_bytes(contents)

    def pipeline_task():
        try:
            outputs = {}
            stages = []

            def update_stage(name, status):
                stage_entry = {"stage": name, "status": status}
                stages.append(stage_entry)
                PIPELINE_PROGRESS[session_id]["stages"] = stages.copy()

            # Step 1: Cleaning
            stage_name = "Cleaning"
            if dataset == "ILC":
                run_script(SCRIPTS_DIR / "clean_ilc_data.py", ["--n", str(n)], verbose=True, stage_name=stage_name)
                outputs["cleaned"] = str(DATA_DIR / "cleaned_ilc.json")
            else:
                run_script(SCRIPTS_DIR / "clean_inabs_sample.py", ["--n", str(n)], verbose=True, stage_name=stage_name)
                outputs["cleaned"] = str(DATA_DIR / "cleaned_inabs.json")
            update_stage(stage_name, "completed")

            # Step 2: Chunking (ILC only)
            if dataset == "ILC":
                stage_name = "Chunking"
                run_script(SCRIPTS_DIR / "chunk_ilc_t5.py", ["--n", str(n)], verbose=True, stage_name=stage_name)
                outputs["chunked"] = str(DATA_DIR / "chunked_ilc.json")
                update_stage(stage_name, "completed")

            # Step 3: Summarization
            stage_name = "Summarization"
            if dataset == "ILC":
                run_script(SCRIPTS_DIR / "t5_ilc.py", ["--n", str(n)], verbose=True, stage_name=stage_name)
                outputs["summary"] = str(DATA_DIR / "t5_ilc_final.json")
            else:
                run_script(SCRIPTS_DIR / "t5_inabs.py", ["--n", str(n)], verbose=True, stage_name=stage_name)
                outputs["summary"] = str(DATA_DIR / "t5_inabs_final.json")
            update_stage(stage_name, "completed")

            # Step 4: Evaluation
            stage_name = "Evaluation"
            run_script(SCRIPTS_DIR / "t5_evaluation.py", ["--dataset", dataset, "--n", str(n)], verbose=True, stage_name=stage_name)
            outputs["evaluation"] = str(DATA_DIR / f"rouge_{dataset.lower()}.json")
            update_stage(stage_name, "completed")

            # Load JSON results
            cleaned_path = DATA_DIR / ("cleaned_ilc.json" if dataset == "ILC" else "cleaned_inabs.json")
            summary_path = DATA_DIR / ("t5_ilc_final.json" if dataset == "ILC" else "t5_inabs_final.json")
            eval_path = DATA_DIR / f"rouge_{dataset.lower()}.json"

            cleaned_json = safe_load_json(cleaned_path)
            summary_json = safe_load_json(summary_path)
            eval_json = safe_load_json(eval_path)

            # Build entries for frontend with per-entry ROUGE
            entries = []
            per_entry_rouge = eval_json.get("per_entry", []) if isinstance(eval_json, dict) else []
            if isinstance(cleaned_json, list) and isinstance(summary_json, list):
                L = min(len(cleaned_json), len(summary_json), n)
                for i in range(L):
                    c, s = cleaned_json[i], summary_json[i]
                    rouge_entry = per_entry_rouge[i] if i < len(per_entry_rouge) else None
                    entries.append(
                        {
                            "original_text": c.get("input_text"),
                            "reference_summary": c.get("summary_text"),
                            "generated_summary": s.get("refined_summary_improved"),
                            "rouge": rouge_entry,
                        }
                    )

            stored_entry_id = PIPELINE_PROGRESS[session_id].get("entry_id")
            if stored_entry_id and stored_entry_id > 0 and stored_entry_id <= len(entries):
                entries = [entries[stored_entry_id - 1]]

            # Avg ROUGE metrics
            avg_rouge = {"rouge1": None, "rouge2": None, "rougeL": None}
            if isinstance(eval_json, dict):
                for k in avg_rouge.keys():
                    avg_rouge[k] = eval_json.get(k) or eval_json.get("scores", {}).get(k)

            PIPELINE_PROGRESS[session_id]["results"] = {
                "avg_rouge": avg_rouge,
                "entries": entries,
                "outputs": outputs,
            }
            PIPELINE_PROGRESS[session_id]["completed"] = True

        except Exception as e:
            PIPELINE_PROGRESS[session_id]["error"] = str(e)
            PIPELINE_PROGRESS[session_id]["completed"] = True

    Thread(target=pipeline_task).start()

    return {"status": "started", "session_id": session_id}

@app.get("/pipeline_status")
def pipeline_status(session_id: str):
    if session_id not in PIPELINE_PROGRESS:
        raise HTTPException(status_code=404, detail="Session not found")
    return PIPELINE_PROGRESS[session_id]
