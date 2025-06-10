# Real-time Food Detection System

A real-time food detection system for automated restaurant billing using YOLO object detection and optical flow tracking. The system detects food items on trays and automatically calculates bills based on detected items.

## Quick Start

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- a webcam

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/korkmaz-arda/food-detection-server
   ```

2. **Install dependencies**
   
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up authentication** (for admin features)
   
   ```bash
   # Generate password hash locally and create .auth_token file with the generated hash
   pip install argon2-cffi
   python -c "from argon2 import PasswordHasher; print(PasswordHasher().hash('PASSWORD_STRING'))" > .auth_token   
   ```

4. **Run the server**
   ```bash
   python server.py
   ```
   The server will start on `http://localhost:8890`

## Usage

### For Restaurants (User Interface)
1. Navigate to `http://localhost:8890/user`
2. Click "Connect" to connect to the detection server
3. Click "Start Detection" and allow camera access
4. Point camera at food trays - items will be detected automatically
5. Click "Checkout" to see the bill

### For Developers (Dev Interface)
1. Navigate to `http://localhost:8890/dev`
2. Access additional features:
   - Confidence scores
   - Detection statistics
   - Detailed logging
   - Position information

### For Administrators (Model Management)
1. Navigate to `http://localhost:8890/models`
2. Upload new detection models
3. Switch between different models
4. Configure detection parameters

## Features

### Real-time Detection
- **YOLO-based detection**: Detects food items and trays in real-time
- **Optical flow tracking**: Tracks items between frames for smooth performance
- **Automatic billing**: Calculates prices based on detected items

### Model Management
- Upload custom YOLO models for different restaurants/menus
- Configure detection confidence thresholds
- Support for custom food lists and pricing

### Multiple Interfaces
- **User Client**: Simplified interface for restaurant staff
- **Dev Client**: Detailed interface for testing and debugging
- **Admin Panel**: Model management and configuration

## API Endpoints

### WebSocket
- `ws://localhost:8890/ws` - Real-time detection stream

### REST API
- `GET /` - Server status and links
- `GET /user` - User interface
- `GET /dev` - Developer interface
- `GET /models` - Model management interface
- `GET /api/models/{type}` - List models (type: 'food' or 'tray')
- `POST /api/models/upload` - Upload new model
- `POST /api/models/activate` - Activate a model
- `DELETE /api/models/{type}/{name}` - Delete a model

## Architecture

```
├── server.py              # FastAPI server with WebSocket support
├── detection.py           # Detection manager with YOLO and optical flow
├── model_manager.py       # Model versioning and management
├── models/                # Detection models
│   ├── food/             # Food detection models
│   └── tray/             # Tray detection models
├── utils/                 # Detection utilities
│   ├── bbox.py           # Bounding box utilities
│   └── detect.py         # YOLO detection functions
├── *_client.html         # Web interfaces
└── models_admin.html     # Admin interface
```
