import uvicorn
import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def main():
    """Run the web application."""
    uvicorn.run(
        "pubvec.web.app:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )

if __name__ == "__main__":
    main() 