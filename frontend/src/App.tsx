import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import AnimatedBackground from './components/AnimatedBackground';
import LandingPage from './components/LandingPage';
import MainApp from './pages/MainApp';
import './style.css';

function App() {
    return (
        <Router>
            <AnimatedBackground />
            <Routes>
                <Route path="/" element={<LandingPage />} />
                <Route path="/app" element={<MainApp />} />
                {/* Flask will handle /overview, /system, /about */}
            </Routes>
        </Router>
    );
}

export default App;
