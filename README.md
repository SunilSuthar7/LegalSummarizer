# Legal Summarizer

A full-stack application for legal document summarization with real-time progress tracking.

## Project Structure

\`\`\`
project-root/
├─ backend/
│   ├─ main.py              # FastAPI backend server
│   ├─ run_server.py        # Easy server startup script
│   ├─ requirements.txt     # Python dependencies
│   └─ scripts/            # Pipeline scripts (add your scripts here)
│       ├─ clean_ilc_data.py
│       ├─ clean_inabs_sample.py
│       ├─ chunk_ilc_t5.py
│       ├─ t5_ilc.py
│       ├─ t5_inabs.py
│       └─ t5_evaluation.py
└─ frontend/
    ├─ index.html          # Main HTML interface
    ├─ style.css           # Styling
    └─ script.js           # Frontend logic
\`\`\`

## Quick Start

### 🚀 Easy Setup (Recommended)

1. **Start the Backend Server:**
   \`\`\`bash
   cd backend
   pip install -r requirements.txt
   python run_server.py
   \`\`\`
   The backend will be available at `http://127.0.0.1:8000`

2. **Start the Frontend Server:**
   Open a new terminal and run:
   \`\`\`bash
   cd frontend
   # Using Live Server extension in VS Code (recommended)
   # Or using Python's built-in server:
   python -m http.server 5500
   \`\`\`
   The frontend will be available at `http://127.0.0.1:5500`

3. **Open your browser** to `http://127.0.0.1:5500` and start using the application!

### 🔧 Manual Setup

#### Backend Setup

1. Navigate to the backend directory:
   \`\`\`bash
   cd backend
   \`\`\`

2. Install Python dependencies:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

3. Add your pipeline scripts to the `backend/scripts/` directory

4. Run the FastAPI server:
   \`\`\`bash
   uvicorn main:app --reload --host 127.0.0.1 --port 8000
   \`\`\`

#### Frontend Setup

1. Navigate to the frontend directory:
   \`\`\`bash
   cd frontend
   \`\`\`

2. Serve the frontend files using any web server:
   \`\`\`bash
   # Using Python's built-in server
   python -m http.server 5500
   
   # Or using Node.js serve
   npx serve -p 5500
   
   # Or use VS Code Live Server extension (port 5500)
   \`\`\`

3. Open your browser to `http://127.0.0.1:5500`

## ⚠️ Troubleshooting

If you see **"405 Method Not Allowed"** errors:
- Make sure the backend server is running on port 8000
- Check that both servers are running simultaneously
- Verify the frontend is accessing `http://127.0.0.1:5500`

If you see **CORS errors**:
- The backend is configured to allow all origins for development
- Make sure you're using the correct ports (frontend: 5500, backend: 8000)

## Features

- **Dataset Selection**: Choose between ILC and IN-ABS datasets
- **Configurable Processing**: Set number of entries and optional specific entry ID
- **File Upload**: Optional JSON file upload for custom data
- **Real-time Progress**: Live progress bars for each pipeline stage
- **Comprehensive Results**: View original text, reference summaries, generated summaries, and ROUGE scores
- **Responsive Design**: Works on desktop and mobile devices

## API Endpoints

- `POST /run_pipeline`: Start the summarization pipeline
- `GET /pipeline_status?session_id=<id>`: Get real-time pipeline status
- `GET /`: Health check endpoint

## Deployment

This application is designed to work on Vercel or any platform that supports both Python backends and static frontend hosting.
\`\`\`

```html file="" isHidden
