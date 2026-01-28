# Malaysian Kuih Recognition - React Frontend

## 🎯 Project Overview

This project has been migrated from a Flask template-based application to a modern React SPA with a stunning landing page, animated backgrounds, and enhanced UI/UX.

![Project Status](https://img.shields.io/badge/Status-In%20Development-yellow)
![React](https://img.shields.io/badge/React-18-blue)
![TypeScript](https://img.shields.io/badge/TypeScript-5-blue)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3-cyan)

---

## 📁 Project Structure

```
MKR-CEKG-lite/
├── frontend/                  # React application
│   ├── src/
│   │   ├── components/        # React components
│   │   │   ├── AnimatedBackground.tsx
│   │   │   ├── LandingPage.tsx
│   │   │   ├── Navigation.tsx
│   │   │   └── ImageUpload.tsx
│   │   ├── pages/            # Page components
│   │   │   └──MainApp.tsx
│   │   ├── services/          # API integration
│   │   │   └── api.ts
│   │   ├── App.tsx            # Main app component
│   │   └── style.css          # Tailwind styles
│   ├── package.json
│   └── tailwind.config.js
├── templates_backup_20260127/ # Backup of original templates
├── app_backup_20260127.py     # Backup of original Flask app
├── app.py                     # Flask backend with CORS
├── requirements.txt
└── README.md

```

---

## 🚀 Quick Start

### Prerequisites

- **Node.js** 18+ and npm
- **Python** 3.8+
- **MongoDB Atlas** account (for database)
- **Gemini API Key** (for AI features)

### 1. Backend Setup (Flask)

```bash
# Install Python dependencies
pip install -r requirements.txt
pip install flask-cors

# Set environment variables
set MONGO_DB_PASSWORD=your_password
set GEMINI_API_KEY=your_api_key

# Run Flask backend
python app.py
```

The Flask API will run on `http://localhost:5000`

### 2. Frontend Setup (React)

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

The React app will run on `http://localhost:5174` (or 5173)

---

## 🎨 Features

### ✅ Implemented

- **Beautiful Landing Page** - Gradient text, floating animations, modern design
- **Animated Background** - Canvas-based gradient blobs with reduced motion support
- **Image Upload** - Drag-and-drop support with file validation
- **CNN Classification** - Real-time kuih recognition
- **Responsive Design** - Mobile, tablet, and desktop optimized
- **Glassmorphism UI** - Modern glass-like effects
- **CORS Enabled** - Flask backend ready for React frontend

### 🚧 In Progress

- Gemini AI Knowledge Card
- Poster Generation Interface
- History Panel
- Complete API integration testing

### 📋 Planned

- Overview, System, About pages (or keep Flask templates)
- Advanced state management
- Production deployment guide

---

## 🛠️ Technology Stack

### Frontend
- **React 18** - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool & dev server
- **Tailwind CSS** - Styling framework
- **React Router** - Client-side routing
- **Axios** - HTTP client

### Backend
- **Flask** - Python web framework
- **TensorFlow/Keras** - CNN model
- **Gemini AI** - Vision & knowledge generation
- **MongoDB Atlas** - Database
- **Redis** (optional) - Job queue for poster generation

---

## 📝 Development Workflow

### Running Both Servers

**Terminal 1 - Flask Backend:**
```bash
python app.py
```

**Terminal 2 - React Frontend:**
```bash
cd frontend
npm run dev
```

### Building for Production

```bash
cd frontend
npm run build
```

Build output will be in `frontend/dist/`

---

## 🎯 API Endpoints

All API endpoints are CORS-enabled for `localhost:5173`, `localhost:5174`, and `localhost:3000`.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Upload image for classification |
| `/gemini-info` | POST | Get AI-generated kuih information |
| `/generate_poster` | POST | Create AI recipe poster |
| `/poster_status/:id` | GET | Check poster generation job status |
| `/poster_quota` | GET | Get poster generation quota |
| `/unlock_poster` | POST | Unlock poster quota with code |
| `/api/history` | GET | Get prediction history |
| `/api/history/:id` | DELETE | Delete history item |
| `/submit_feedback` | POST | Submit feedback on prediction |

---

## 🔒 Security Notes

- **CORS** - Currently allows localhost origins only
- **File Upload** - Max 16MB, validated extensions
- **Environment Variables** - Never commit `.env` files
- **Rate Limiting** - Poster generation quota system in place

---

## 🐛 Troubleshooting

### Frontend won't start
- Make sure you're in the `frontend/` directory
- Run `npm install` to ensure all dependencies are installed
- Check that port 5173/5174 isn't already in use

### Backend CORS errors
- Verify Flask-CORS is installed: `pip install flask-cors`
- Check that the Flask server is running
- Ensure the frontend URL matches the CORS configuration

### API calls failing
- Verify the Flask backend is running on port 5000
- Check `.env` file has `VITE_API_URL=http://localhost:5000`
- Check browser console for detailed error messages

---

## 📦 Backup & Recovery

**Backups created:**
- `templates_backup_20260127/` - Original HTML templates
- `app_backup_20260127.py` - Original Flask app (before CORS)

To revert to the original system, simply use these backups.

---

## 🔮 Future Enhancements

- [ ] Add unit tests (Jest + React Testing Library)
- [ ] Implement Progressive Web App (PWA) features
- [ ] Add internationalization (i18n) support
- [ ] Optimize image loading with lazy loading
- [ ] Add error boundaries for better error handling
- [ ] Implement analytics tracking

---

## 👤 Author

**Final Year Project** - Malaysian Kuih Recognition System
CNN Architecture Variants + Gemini AI

---

## 📄 License

All rights reserved © 2025

---

## 🙏 Acknowledgments

- TensorFlow/Keras for the CNN model framework
- Google Gemini AI for vision and knowledge generation
- Tailwind CSS for the design system
- Vite team for the excellent build tool

