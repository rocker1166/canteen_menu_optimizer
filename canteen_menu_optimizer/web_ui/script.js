// API Configuration
const API_BASE_URL = 'http://127.0.0.1:8000';

// DOM Elements
const apiStatusElement = document.getElementById('api-status');
const predictionForm = document.getElementById('prediction-form');
const predictionResult = document.getElementById('prediction-result');
const menuItemsContainer = document.getElementById('menu-items');
const modelInfoContainer = document.getElementById('model-info');
const itemSelect = document.getElementById('item-id');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    // Set today's date as default
    document.getElementById('date').value = new Date().toISOString().split('T')[0];
    
    // Initialize all sections
    checkApiStatus();
    loadMenuItems();
    loadModelInfo();
    
    // Set up form submission
    predictionForm.addEventListener('submit', handlePredictionSubmit);
});

// Check API Status
async function checkApiStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            const data = await response.json();
            updateApiStatus('healthy', `API is healthy (${data.model_version})`);
        } else {
            updateApiStatus('error', 'API is not responding');
        }
    } catch (error) {
        updateApiStatus('error', 'Failed to connect to API');
        console.error('API Status Error:', error);
    }
}

// Update API Status Display
function updateApiStatus(status, message) {
    const statusDot = apiStatusElement.querySelector('.status-dot');
    const statusText = apiStatusElement.querySelector('span:last-child');
    
    statusDot.className = `status-dot ${status}`;
    statusText.textContent = message;
}

// Load Menu Items
async function loadMenuItems() {
    try {
        const response = await fetch(`${API_BASE_URL}/menu-items`);
        if (response.ok) {
            const data = await response.json();
            displayMenuItems(data.menu_items);
            populateItemSelect(data.menu_items);
        } else {
            showError('Failed to load menu items');
        }
    } catch (error) {
        showError('Error loading menu items: ' + error.message);
        console.error('Menu Items Error:', error);
    }
}

// Display Menu Items
function displayMenuItems(menuItems) {
    menuItemsContainer.innerHTML = '';
    
    menuItems.forEach(item => {
        const menuItemElement = document.createElement('div');
        menuItemElement.className = 'menu-item';
        menuItemElement.innerHTML = `
            <h4>${item.name}</h4>
            <div class="price">₹${item.price}</div>
            <div class="item-id">${item.id}</div>
        `;
        menuItemsContainer.appendChild(menuItemElement);
    });
}

// Populate Item Select Dropdown
function populateItemSelect(menuItems) {
    // Clear existing options except the first one
    itemSelect.innerHTML = '<option value="">Select an item...</option>';
    
    menuItems.forEach(item => {
        const option = document.createElement('option');
        option.value = item.id;
        option.textContent = item.name;
        itemSelect.appendChild(option);
    });
}

// Load Model Information
async function loadModelInfo() {
    try {
        const response = await fetch(`${API_BASE_URL}/model-info`);
        if (response.ok) {
            const data = await response.json();
            displayModelInfo(data);
        } else {
            showError('Failed to load model information');
        }
    } catch (error) {
        showError('Error loading model info: ' + error.message);
        console.error('Model Info Error:', error);
    }
}

// Display Model Information
function displayModelInfo(modelData) {
    modelInfoContainer.innerHTML = '';
    
    // ML Model Card
    const mlModelCard = document.createElement('div');
    mlModelCard.className = 'model-card';
    mlModelCard.innerHTML = `
        <h4><i class="fas fa-brain"></i> Machine Learning Model</h4>
        <div class="model-details">
            <div class="model-detail">
                <span class="label">Type:</span>
                <span class="value">${modelData.ml_model.type}</span>
            </div>
            <div class="model-detail">
                <span class="label">Features:</span>
                <span class="value">${modelData.ml_model.features}</span>
            </div>
            <div class="model-detail">
                <span class="label">RMSE:</span>
                <span class="value">${modelData.ml_model.performance.rmse}</span>
            </div>
            <div class="model-detail">
                <span class="label">MAE:</span>
                <span class="value">${modelData.ml_model.performance.mae}</span>
            </div>
            <div class="model-detail">
                <span class="label">R² Score:</span>
                <span class="value">${modelData.ml_model.performance.r2_score}</span>
            </div>
        </div>
    `;
    
    // RL Model Card
    const rlModelCard = document.createElement('div');
    rlModelCard.className = 'model-card';
    rlModelCard.innerHTML = `
        <h4><i class="fas fa-robot"></i> Reinforcement Learning Model</h4>
        <div class="model-details">
            <div class="model-detail">
                <span class="label">Type:</span>
                <span class="value">${modelData.rl_model.type}</span>
            </div>
            <div class="model-detail">
                <span class="label">State Size:</span>
                <span class="value">${modelData.rl_model.state_size}</span>
            </div>
            <div class="model-detail">
                <span class="label">Action Size:</span>
                <span class="value">${modelData.rl_model.action_size}</span>
            </div>
            <div class="model-detail">
                <span class="label">Episodes Trained:</span>
                <span class="value">${modelData.rl_model.episodes_trained}</span>
            </div>
        </div>
    `;
    
    // Key Features Card
    const featuresCard = document.createElement('div');
    featuresCard.className = 'model-card';
    featuresCard.innerHTML = `
        <h4><i class="fas fa-key"></i> Key Features</h4>
        <ul class="feature-list">
            ${modelData.key_features.map(feature => `<li>${feature}</li>`).join('')}
        </ul>
    `;
    
    modelInfoContainer.appendChild(mlModelCard);
    modelInfoContainer.appendChild(rlModelCard);
    modelInfoContainer.appendChild(featuresCard);
}

// Handle Prediction Form Submission
async function handlePredictionSubmit(event) {
    event.preventDefault();
    
    const formData = new FormData(predictionForm);
    const requestData = {
        date: formData.get('date'),
        item_id: formData.get('item_id'),
        current_stock: formData.get('current_stock') ? parseInt(formData.get('current_stock')) : null,
        rainfall_today: formData.get('rainfall_today') ? parseFloat(formData.get('rainfall_today')) : null,
        student_count: formData.get('student_count') ? parseInt(formData.get('student_count')) : null,
        event_today: parseInt(formData.get('event_today'))
    };
    
    // Remove null values
    Object.keys(requestData).forEach(key => {
        if (requestData[key] === null || requestData[key] === '') {
            delete requestData[key];
        }
    });
    
    await getPrediction(requestData);
}

// Get Prediction from API
async function getPrediction(requestData) {
    const submitButton = document.querySelector('.btn-predict');
    const originalText = submitButton.innerHTML;
    
    // Show loading state
    submitButton.innerHTML = '<div class="loading"></div> Getting Prediction...';
    submitButton.disabled = true;
    
    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });
        
        if (response.ok) {
            const data = await response.json();
            displayPredictionResult(data);
            showSuccess('Prediction generated successfully!');
        } else {
            const errorData = await response.json();
            showError(`Prediction failed: ${errorData.detail}`);
        }
    } catch (error) {
        showError('Error getting prediction: ' + error.message);
        console.error('Prediction Error:', error);
    } finally {
        // Restore button state
        submitButton.innerHTML = originalText;
        submitButton.disabled = false;
    }
}

// Display Prediction Result
function displayPredictionResult(data) {
    document.getElementById('result-item').textContent = data.item_id.replace('_', ' ').toUpperCase();
    document.getElementById('result-quantity').textContent = `${data.predicted_quantity} units`;
    document.getElementById('result-version').textContent = data.model_version;
    
    predictionResult.style.display = 'block';
    predictionResult.scrollIntoView({ behavior: 'smooth' });
}

// Show Error Message
function showError(message) {
    removeMessages();
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.innerHTML = `<i class="fas fa-exclamation-triangle"></i> ${message}`;
    predictionForm.appendChild(errorDiv);
    
    setTimeout(() => {
        errorDiv.remove();
    }, 5000);
}

// Show Success Message
function showSuccess(message) {
    removeMessages();
    const successDiv = document.createElement('div');
    successDiv.className = 'success-message';
    successDiv.innerHTML = `<i class="fas fa-check-circle"></i> ${message}`;
    predictionForm.appendChild(successDiv);
    
    setTimeout(() => {
        successDiv.remove();
    }, 3000);
}

// Remove existing messages
function removeMessages() {
    const existingMessages = predictionForm.querySelectorAll('.error-message, .success-message');
    existingMessages.forEach(msg => msg.remove());
}

// Utility Functions
function formatDate(date) {
    return new Date(date).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });
}

function capitalizeWords(str) {
    return str.replace(/\w\S*/g, (txt) => {
        return txt.charAt(0).toUpperCase() + txt.substr(1).toLowerCase();
    });
}

// Refresh data periodically
setInterval(() => {
    checkApiStatus();
}, 30000); // Check API status every 30 seconds
