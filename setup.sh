#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "🚀 Starting Maritime AI Project Setup..."

# --- Prerequisites Check (Optional but Recommended) ---
echo "🔎 Checking prerequisites..."

# Check for Python 3
if ! command -v python3 &> /dev/null
then
    echo "❌ Error: Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi
echo "✅ Python 3 found."

# Check for pip3
if ! command -v pip3 &> /dev/null
then
    echo "❌ Error: pip3 is not installed. Please install pip for Python 3 (e.g., 'sudo apt install python3-pip' or 'brew install python') and try again."
    exit 1
fi
echo "✅ pip3 found."

# Check for Node.js and npm
if ! command -v node &> /dev/null || ! command -v npm &> /dev/null
then
    echo "❌ Error: Node.js or npm is not installed. Please install Node.js (which includes npm) and try again (https://nodejs.org/)."
    exit 1
fi
echo "✅ Node.js and npm found."


# --- Backend Setup (Python) ---
echo "🐍 Setting up Python backend in 'maritime-tracking'..."

# Navigate to the backend directory
cd maritime-tracking

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: 'maritime-tracking/requirements.txt' not found. Cannot install Python dependencies."
    cd .. # Go back to root before exiting
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "   Creating Python virtual environment 'venv'..."
    python3 -m venv venv
else
    echo "   Python virtual environment 'venv' already exists."
fi

# Activate virtual environment and install dependencies
echo "   Activating virtual environment and installing Python dependencies..."
source venv/bin/activate
pip3 install --upgrade pip # Upgrade pip first
pip3 install -r requirements.txt
echo "   Deactivating virtual environment for now."
deactivate # Deactivate after install to not leave the user's shell activated

# Navigate back to the root directory
cd ..
echo "✅ Python backend setup complete."


# --- Frontend Setup (Node.js) ---
echo "💻 Setting up Node.js frontend..."

# Check if package.json exists
if [ ! -f "package.json" ]; then
    echo "❌ Error: 'package.json' not found in the root directory. Cannot install Node.js dependencies."
    exit 1
fi

# Install Node.js dependencies
echo "   Installing Node.js dependencies using npm..."
npm install
echo "✅ Node.js frontend setup complete."


# --- Final Instructions ---
echo ""
echo "🎉 Setup finished successfully!"
echo ""
echo "Next Steps:"
echo "1. Prepare your AIS data (see README.md section 'Data')."
echo "2. Activate the Python environment when running backend scripts:"
echo "   cd maritime-tracking"
echo "   source venv/bin/activate"
echo "   # ... run your python scripts (e.g., for training or API) ..."
echo "   deactivate"
echo "3. Run the frontend development server (from the project root directory):"
echo "   npm run dev"
echo "4. Open your browser to http://localhost:3000 (or the specified port)."
echo ""

exit 0 