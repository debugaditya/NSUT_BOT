NSUTBot 🤖NSUTBot is an intelligent, RAG-based (Retrieval-Augmented Generation) chatbot designed to assist students of Netaji Subhas University of Technology (NSUT). It provides instant, context-aware answers regarding coursework, syllabus, coding problems, and university guidelines by retrieving information from a dedicated knowledge base.The project features a polished, Gemini-inspired UI built with React and a robust Python backend handling vector embeddings and AI inference.✨ Features🧠 Intelligent Q&A: Context-aware responses based on university data using RAG.⚡ Streaming Responses: Real-time, character-by-character typing effect for a natural feel.📝 Rich Text Support: Renders Markdown, LaTeX (Math equations), and Syntax Highlighted Code blocks perfectly.📎 File Attachments: Users can upload documents (PDF/Text) for specific, context-aware queries.🎨 Modern UI/UX: A responsive, dark-mode interface with a "floating capsule" input bar, similar to Google Gemini.🗄️ Vector Database: Uses a local vector store (vector_db) for efficient information retrieval.🛠️ Tech StackFrontendReact (Vite): Fast, modern UI development.Framer Motion: Smooth animations for sidebars and transitions.Lucide React: Clean, consistent iconography.React Markdown & Katex: For rendering complex scientific and mathematical content.BackendPython: Core server logic.LangChain: For managing LLM chains and retrieval.Vector DB: Stores document embeddings for the RAG system.Flask/FastAPI: (Implied) API handling.📂 Project StructureNSUTBOT/
├── backend/                # Python Server
│   ├── vector_db/          # Embeddings storage
│   ├── uploads/            # Temporary storage for attached files
│   ├── nsutbot.py          # Main application entry point
│   ├── requirements.txt    # Python dependencies
│   └── .env                # Backend secrets (API Keys)
│
├── frontend/               # React Client
│   ├── src/                # UI Components and Logic
│   ├── package.json        # Node dependencies
│   ├── vite.config.js      # Build configuration
│   └── .env                # Frontend config
│
└── README.md
🚀 Getting StartedFollow these steps to set up the project locally on your machine.PrerequisitesNode.js (v16 or higher)Python (v3.9 or higher)Git1. Clone the Repositorygit clone [https://github.com/debugaditya/nsutbot.git](https://github.com/debugaditya/nsutbot.git)
cd nsutbot
2. Backend Setup 🐍Navigate to the backend folder and set up the virtual environment.cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
Configuration:Create a .env file in the backend/ directory and add your necessary API keys (e.g., Gemini, OpenAI, or others):LLM_API_KEY=your_api_key_here
PORT=3000
Run the Server:python nsutbot.py
The backend should now be running on http://localhost:3000 (or your configured port).3. Frontend Setup ⚛️Open a new terminal, navigate to the frontend folder, and install dependencies.cd frontend

# Install dependencies
npm install
Run the Development Server:npm run dev
The frontend will launch, usually accessible at http://localhost:5173.💡 UsageEnsure both the Backend and Frontend terminals are running.Open your browser to the frontend URL.Chat: Type a question in the floating input bar at the bottom.Attach Files: Click the + icon to upload a document for context-specific questions.Math/Code: Try asking for a mathematical formula or a code snippet to see the rendering capabilities.🤝 ContributingContributions are welcome!Fork the repository.Create a feature branch (git checkout -b feature/AmazingFeature).Commit your changes (git commit -m 'Add some AmazingFeature').Push to the branch (git push origin feature/AmazingFeature).Open a Pull Request.👤 AuthorAditya BarmolaGitHub: @debugaditya📜 LicenseDistributed under the MIT License. See LICENSE for more information.