// --- Theme Toggle Logic ---
const themeToggleBtn = document.getElementById('theme-toggle');

// Determine system and storage preferences
const savedTheme = localStorage.getItem('theme');
const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
const initialTheme = savedTheme || (systemPrefersDark ? 'dark' : 'light');

// Apply the initial theme immediately on load
setTheme(initialTheme);

// Toggle theme on user click
themeToggleBtn.addEventListener('click', () => {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    setTheme(newTheme);
});

function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
    themeToggleBtn.textContent = theme === 'dark' ? '☀️' : '🌙';
}

// --- AI Course Creator Logic ---
const createForm = document.getElementById('create-form');
const topicInput = document.getElementById('topic-input');
const createButton = document.getElementById('create-button');
const progressContainer = document.getElementById('progress-container');
const statusText = document.getElementById('status-text');

// Generate a random session ID for this browser session
const sessionId = 'session-' + Math.random().toString(36).substring(2, 15);

function showProgress() {
    createForm.classList.add('hidden'); // Optionally hide form, or just disable
    topicInput.disabled = true;
    createButton.disabled = true;
    createButton.innerHTML = 'Building...';
    progressContainer.classList.remove('hidden');
}

function updateStatus(text) {
    statusText.textContent = text;
    
    // Simple logic to highlight steps based on text content
    document.querySelectorAll('.step').forEach(s => s.classList.remove('active'));
    
    if (text.toLowerCase().includes('research')) {
        document.getElementById('step-researcher').classList.add('active');
    } else if (text.toLowerCase().includes('judge') || text.toLowerCase().includes('evaluating')) {
        document.getElementById('step-judge').classList.add('active');
    } else if (text.toLowerCase().includes('writ') || text.toLowerCase().includes('build')) {
        document.getElementById('step-builder').classList.add('active');
    }
}

createForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const topic = topicInput.value.trim();
    if (!topic) return;

    showProgress();

    try {
        const response = await fetch('/api/chat_stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: `Create a comprehensive course on: ${topic}`,
                session_id: sessionId
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (!line.trim()) continue;
                try {
                    const data = JSON.parse(line);
                    if (data.type === 'progress') {
                        updateStatus(data.text);
                    } else if (data.type === 'result') {
                        // Save result and redirect
                        localStorage.setItem('currentCourse', data.text);
                        window.location.href = '/course.html';
                        return;
                    }
                } catch (e) {
                    console.error('Error parsing JSON:', e, line);
                }
            }
        }

    } catch (error) {
        console.error('Error:', error);
        statusText.textContent = 'Something went wrong. Please refresh and try again.';
        // Re-enable form if needed, or just let them refresh
    }
});
