const runBtn = document.getElementById("runBtn")
const datasetEl = document.getElementById("dataset")
const numEntriesEl = document.getElementById("numEntries")
const entryIdEl = document.getElementById("entryId")
const fileInput = document.getElementById("fileInput")

const errorEl = document.getElementById("error")
const resultsEl = document.getElementById("results")

const fillCleaning = document.getElementById("fillCleaning")
const fillChunking = document.getElementById("fillChunking")
const fillSummarization = document.getElementById("fillSummarization")
const fillEvaluation = document.getElementById("fillEvaluation")

let pollInterval

const API_BASE =
  window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
    ? "http://127.0.0.1:8000"
    : window.location.origin

function resetProgress() {
  fillCleaning.style.width = "0%"
  fillChunking.style.width = "0%"
  fillSummarization.style.width = "0%"
  fillEvaluation.style.width = "0%"
}

function updateProgress(stages) {
  stages.forEach((stage) => {
    const stageName = stage.stage.toLowerCase()
    const width = stage.status === "completed" ? "100%" : "50%"

    if (stageName.includes("clean")) fillCleaning.style.width = width
    if (stageName.includes("chunk")) fillChunking.style.width = width
    if (stageName.includes("summar")) fillSummarization.style.width = width
    if (stageName.includes("evalu")) fillEvaluation.style.width = width
  })
}

function setRunningState(isRunning) {
  runBtn.disabled = isRunning
  runBtn.innerHTML = isRunning
    ? '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2"/><path d="M12 6v6l4 2" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>Processing...'
    : '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><polygon points="5,3 19,12 5,21" fill="currentColor"/></svg>Run Pipeline'

  // Disable form controls while running
  datasetEl.disabled = isRunning
  numEntriesEl.disabled = isRunning
  entryIdEl.disabled = isRunning
  fileInput.disabled = isRunning
}

runBtn.onclick = async () => {
  // Reset UI state
  errorEl.style.display = "none"
  resultsEl.innerHTML = ""
  resetProgress()
  setRunningState(true)

  const formData = new FormData()
  formData.append("dataset", datasetEl.value)

  const specificEntryId = entryIdEl.value.trim()
  if (specificEntryId && Number.parseInt(specificEntryId) > 0) {
    // If specific entry ID is provided, use it and ignore number of entries
    formData.append("n", Number.parseInt(specificEntryId))
    formData.append("entry_id", Number.parseInt(specificEntryId))
  } else {
    // Otherwise use the selected number of entries
    formData.append("n", Number(numEntriesEl.value))
  }

  if (fileInput.files.length > 0) formData.append("file", fileInput.files[0])

  try {
    const res = await fetch(`${API_BASE}/run_pipeline`, {
      method: "POST",
      body: formData,
    })
    const data = await res.json()
    if (!res.ok) throw new Error(data?.error || res.statusText)

    const sessionId = data.session_id

    // Polling every 1s
    pollInterval = setInterval(async () => {
      try {
        const statusRes = await fetch(`${API_BASE}/pipeline_status?session_id=${sessionId}`)
        const statusData = await statusRes.json()

        // Update progress bars
        if (statusData.stages) {
          updateProgress(statusData.stages)
        }

        if (statusData.completed) {
          clearInterval(pollInterval)
          setRunningState(false)

          const results = statusData.results
          if (results) {
            let entries = results.entries || []
            const specificId = Number.parseInt(entryIdEl.value.trim())
            if (specificId && specificId > 0 && specificId <= entries.length) {
              entries = [entries[specificId - 1]]
            }

            if (entries.length > 0) {
              resultsEl.innerHTML = "<h3>Legal Document Summaries</h3>"
              entries.forEach((entry, idx) => {
                resultsEl.innerHTML += `
                  <div class="entry-card">
                    <div class="entry-header">
                      Document ${idx + 1}
                    </div>
                    <div class="entry-content">
                      <div class="text-section">
                        <strong>Original Legal Text:</strong>
                        <div class="text-content original">${entry.original_text ?? "N/A"}</div>
                      </div>
                      
                      <div class="text-section">
                        <strong>Reference Summary:</strong>
                        <div class="text-content reference">${entry.reference_summary ?? "N/A"}</div>
                      </div>
                      
                      <div class="text-section">
                        <strong>AI Generated Summary:</strong>
                        <div class="text-content generated">${entry.generated_summary ?? "N/A"}</div>
                      </div>
                    </div>
                  </div>
                `
              })
            } else {
              resultsEl.innerHTML += "<p>No entries available.</p>"
            }
          }

          if (statusData.error) {
            errorEl.style.display = "block"
            errorEl.textContent = "Pipeline Error: " + statusData.error
          }
        }
      } catch (e) {
        clearInterval(pollInterval)
        setRunningState(false)
        errorEl.style.display = "block"
        errorEl.textContent = "Error fetching pipeline status: " + e.message
      }
    }, 1000)
  } catch (err) {
    setRunningState(false)
    errorEl.style.display = "block"
    errorEl.textContent = "Error: " + err.message
  }
}

window.addEventListener("beforeunload", () => {
  if (pollInterval) {
    clearInterval(pollInterval)
  }
})
