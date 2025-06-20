# Enhanced Canteen Menu Optimizer - Web UI

A modern, responsive web interface for the Enhanced Canteen Menu Optimizer API.

## Features

- **Real-time API Status**: Monitor the health of your FastAPI backend
- **Interactive Prediction Form**: Get AI-powered quantity predictions with optional parameters
- **Menu Items Display**: View all available menu items with pricing
- **Model Information**: Detailed information about ML and RL models
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Modern UI**: Clean, professional interface with smooth animations

## Files

- `index.html` - Main HTML structure
- `style.css` - Modern CSS styling with responsive design
- `script.js` - JavaScript functionality for API interactions

## Usage

1. Make sure your FastAPI server is running:
   ```bash
   python -m uvicorn src.enhanced_api_backend:app --reload --host 127.0.0.1 --port 8000
   ```

2. Open `index.html` in your web browser

3. The web UI will automatically:
   - Check API status
   - Load menu items
   - Load model information

4. Use the prediction form to get quantity predictions by:
   - Selecting a date
   - Choosing a menu item
   - Optionally providing additional context (stock, student count, rainfall, events)

## API Endpoints Used

- `GET /health` - API health check
- `GET /menu-items` - Available menu items
- `GET /model-info` - ML/RL model information
- `POST /predict` - Get quantity prediction

## Browser Compatibility

- Chrome 60+
- Firefox 55+
- Safari 12+
- Edge 79+

## Customization

You can easily customize the UI by modifying:
- Colors and themes in `style.css`
- API base URL in `script.js`
- Layout and content in `index.html`
