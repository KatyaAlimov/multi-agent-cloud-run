# Multi-Agent Cloud Run System

This is the **Basic Version** of the Multi-Agent system developed during the Google Cloud AI Roadshow. This project demonstrates an agentic workflow using the Agent-to-Agent (A2A) protocol, featuring a Researcher, a Judge, and an Orchestrator.

> [!IMPORTANT]
> ### 🔐 Accessing the Advanced Version
> This repository contains the **Basic Implementation**. For access to the **Advanced Branch** (which includes optimized RAG logic, enhanced security templates, and multi-region deployment configurations), please follow these steps:
> 
> 1. **Open an Access Request:** Navigate to the [Issues](../../issues) tab of this repository.
> 2. **Use the Template:** Create a new issue with the title `[Access Request] Advanced Branch - YOUR_GITHUB_USERNAME`.
> 3. **Verification:** In the issue description, briefly state your project requirement.
> 
> *Once your request is approved, you will be added as a collaborator and can switch branches using:*
> `git fetch origin && git checkout advanced`

---

## 🛠 Prerequisites

- A Google Cloud Project with Billing enabled.
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed (if running locally).
- Python 3.10 or higher.

## 📥 How to Download and Setup

To get this project running in your own environment, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/multi-agent-cloud-run.git
   cd multi-agent-cloud-run
2. Initialize Environment Variables:
    echo "PROJECT_ID=$(gcloud config get-value project)" > .env
3. Install Dependencies:
    pip install -r requirements.txt
How to Run Locally
To start all sub-agents and the frontend simultaneously, use the provided orchestration script:

Ensure the script is executable:

chmod +x run_local.sh


Launch the system:

./run_local.sh


Access the App:

If using Cloud Shell, click the Web Preview button and select Port 8080 (or the port defined in your frontend config).
If running locally, open http://localhost:8080 in your browser.
🛑 How to Stop the System
To stop all background agent processes, run:

pkill -f python