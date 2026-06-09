/**
 * Gender Bias Detection - Frontend JavaScript
 * Handles text input, API calls, and result display
 */

// DOM Elements
const textInput = document.getElementById('textInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');
const charCount = document.getElementById('charCount');
const loadingIndicator = document.getElementById('loadingIndicator');
const errorMessage = document.getElementById('errorMessage');
const resultsSection = document.getElementById('resultsSection');
const predictedLabel = document.getElementById('predictedLabel');
const confidenceBadge = document.getElementById('confidenceBadge');
const resultDescription = document.getElementById('resultDescription');
const scoresGrid = document.getElementById('scoresGrid');
const inputTextDisplay = document.getElementById('inputTextDisplay');

// Label color mapping
const labelColors = {
    'GB-ATTACK': '#e74c3c',
    'GB-NORMATIVE': '#f39c12',
    'GB-SEX': '#c0392b',
    'neutral': '#27ae60',
    'meta_counter': '#3498db',
    'gendered_insult': '#8e44ad'
};

/**
 * Update character count as user types
 */
textInput.addEventListener('input', () => {
    charCount.textContent = textInput.value.length;
});

/**
 * Clear all input and results
 */
clearBtn.addEventListener('click', () => {
    textInput.value = '';
    charCount.textContent = '0';
    resultsSection.classList.add('hidden');
    errorMessage.classList.add('hidden');
    textInput.focus();
});

/**
 * Handle analyze button click
 */
analyzeBtn.addEventListener('click', analyzeText);

/**
 * Allow Enter+Ctrl/Cmd to submit
 */
textInput.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        analyzeText();
    }
});

/**
 * Main function to analyze text via API
 */
async function analyzeText() {
    const text = textInput.value.trim();

    // Validation
    if (!text) {
        showError('Please enter some text to analyze.');
        return;
    }

    if (text.length < 5) {
        showError('Please enter at least 5 characters.');
        return;
    }

    // Show loading, hide results/errors
    loadingIndicator.classList.remove('hidden');
    resultsSection.classList.add('hidden');
    errorMessage.classList.add('hidden');

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();

        if (data.error) {
            showError(data.error);
        } else {
            displayResults(data);
        }
    } catch (error) {
        showError(`Error analyzing text: ${error.message}`);
        console.error('Error:', error);
    } finally {
        loadingIndicator.classList.add('hidden');
    }
}

/**
 * Display error message
 */
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove('hidden');
    resultsSection.classList.add('hidden');
}

/**
 * Display prediction results
 */
function displayResults(data) {
    // Update primary prediction
    const label = data.predicted_label;
    const confidence = (data.confidence * 100).toFixed(2);

    predictedLabel.textContent = label;
    predictedLabel.style.borderColor = labelColors[label] || '#3498db';
    predictedLabel.style.color = labelColors[label] || '#3498db';
    
    confidenceBadge.textContent = `${confidence}%`;
    resultDescription.textContent = data.description;

    // Display input text
    inputTextDisplay.textContent = data.input_text;

    // Display confidence scores for all categories
    displayScores(data.all_scores);

    // Show results section with animation
    resultsSection.classList.remove('hidden');
    errorMessage.classList.add('hidden');

    // Scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

/**
 * Display confidence scores for all categories
 */
function displayScores(scores) {
    scoresGrid.innerHTML = ''; // Clear previous scores

    // Sort scores by value (descending)
    const sortedScores = Object.entries(scores)
        .sort(([, a], [, b]) => b - a);

    sortedScores.forEach(([label, score]) => {
        const percentage = (score * 100).toFixed(1);
        const color = labelColors[label] || '#3498db';

        const scoreItem = document.createElement('div');
        scoreItem.className = 'score-item';
        scoreItem.innerHTML = `
            <div class="score-label">${escapeHtml(label)}</div>
            <div class="score-bar-container">
                <div class="score-bar" style="width: ${percentage}%; background: linear-gradient(90deg, ${color} 0%, ${adjustBrightness(color, -20)} 100%);">
                    <span class="score-percentage">${percentage}%</span>
                </div>
            </div>
        `;

        // Animate bar after a slight delay
        setTimeout(() => {
            const bar = scoreItem.querySelector('.score-bar');
            bar.style.width = `${percentage}%`;
        }, 50);

        scoresGrid.appendChild(scoreItem);
    });
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

/**
 * Adjust color brightness (helper for gradient effects)
 */
function adjustBrightness(color, percent) {
    const num = parseInt(color.replace('#', ''), 16);
    const amt = Math.round(2.55 * percent);
    const R = Math.min(255, (num >> 16) + amt);
    const G = Math.min(255, (num >> 8 & 0x00FF) + amt);
    const B = Math.min(255, (num & 0x0000FF) + amt);
    return '#' + (0x1000000 + (R < 255 ? R < 1 ? 0 : R : 255) * 0x10000 +
        (G < 255 ? G < 1 ? 0 : G : 255) * 0x100 +
        (B < 255 ? B < 1 ? 0 : B : 255))
        .toString(16).slice(1);
}

/**
 * Focus on text input when page loads
 */
document.addEventListener('DOMContentLoaded', () => {
    textInput.focus();
});
