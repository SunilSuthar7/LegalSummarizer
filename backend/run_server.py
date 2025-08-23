import uvicorn
import sys
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

if __name__ == "__main__":
    print("🚀 Starting Legal Summarizer Backend Server...")
    print("📍 Backend will be available at: http://127.0.0.1:8000")
    print("📍 Frontend should be served at: http://127.0.0.1:5500 or http://localhost:5500")
    print("📍 Make sure to start your frontend server separately!")
    print("\n" + "="*50)
    
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
