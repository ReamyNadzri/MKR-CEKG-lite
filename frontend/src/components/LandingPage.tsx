import { useState } from 'react';
// import { useNavigate } from 'react-router-dom';
import ImageUpload from './ImageUpload';
import AIAnalysisHub from './AIAnalysisHub'; // Unified Hub
import HistoryPanel from './HistoryPanel';
import KuihListing from './KuihListing';
import IntroSignature from './IntroSignature';
import { uploadImage, type PredictionResponse } from '../services/api';

const LandingPage = () => {
    // const navigate = useNavigate(); // Removed unused hook
    const [introFinished, setIntroFinished] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [predictionResult, setPredictionResult] = useState<PredictionResponse | null>(null);
    const [historyOpen, setHistoryOpen] = useState(false);

    // Calorie Calculator State
    const [pieces, setPieces] = useState(1);

    // Feedback State
    const [feedbackSent, setFeedbackSent] = useState(false);
    const [showCorrection, setShowCorrection] = useState(false);
    const [actualKuih, setActualKuih] = useState('');

    const calculateTotalCalories = (baseCalories: number | string) => {
        if (typeof baseCalories === 'string' && baseCalories.includes('-')) {
            const parts = baseCalories.split('-');
            const avg = (parseInt(parts[0]) + parseInt(parts[1])) / 2;
            return Math.round(avg * pieces);
        }
        const cal = typeof baseCalories === 'string' ? parseInt(baseCalories) : baseCalories;
        return isNaN(cal) ? 'N/A' : Math.round(cal * pieces);
    };

    const handleFeedback = async (isCorrect: boolean) => {
        if (!predictionResult) return;
        try {
            // In a real app, you would send this to the backend
            // await api.post('/submit_feedback', { ... });
            console.log('Feedback submitted:', isCorrect);
            if (!isCorrect) {
                setShowCorrection(true);
            } else {
                setFeedbackSent(true);
            }
        } catch (e) {
            console.error('Error submitting feedback:', e);
        }
    };

    const submitCorrection = async () => {
        if (!actualKuih) return;
        console.log('Correction submitted:', actualKuih);
        setFeedbackSent(true);
        setShowCorrection(false);
    };

    const scrollToApp = () => {
        const appSection = document.getElementById('app-section');
        if (appSection) {
            appSection.scrollIntoView({ behavior: 'smooth' });
        }
    };

    const handleImageUpload = async (file: File) => {
        console.log('handleImageUpload called with file:', file);
        setIsLoading(true);
        setError(null);

        const formData = new FormData();
        formData.append('file', file);

        console.log('Sending request to backend...');
        try {
            const response = await uploadImage(formData);
            console.log('Response received:', response);
            setPredictionResult(response.data);
        } catch (err: any) {
            console.error('Upload error:', err);
            setError(err.response?.data?.error || 'Failed to analyze image. Please try again.');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="min-h-screen">
            {/* Intro Animation */}
            {!introFinished && (
                <IntroSignature
                    onComplete={() => setIntroFinished(true)}
                    position={{ x: 0, y: 0 }}
                    scale={1.5}
                    strokeWidth={2}
                />
            )}
            {/* Hero Section */}
            <div className="min-h-screen flex flex-col items-center justify-center px-4 sm:px-6 lg:px-8 relative overflow-hidden">
                {/* Top Navigation */}
                <nav className="absolute top-0 left-0 right-0 flex justify-between items-center px-6 py-6 z-10">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-600 to-fuchsia-600 flex items-center justify-center text-white shadow-lg text-xl">
                            🥟
                        </div>
                        <span className="font-serif font-bold text-lg text-slate-900">KuihAI</span>
                    </div>
                    <div className="flex items-center gap-4">
                        <button
                            onClick={scrollToApp}
                            className="text-sm font-medium text-slate-700 hover:text-purple-600 transition-colors"
                        >
                            Recognition
                        </button>
                        <a href="/overview" className="text-sm font-medium text-slate-700 hover:text-purple-600 transition-colors">
                            Overview
                        </a>
                        <button
                            onClick={() => setHistoryOpen(true)}
                            className="text-sm font-medium text-slate-700 hover:text-purple-600 transition-colors flex items-center gap-2"
                        >
                            <i className="fa-solid fa-clock-rotate-left"></i>
                            History
                        </button>
                    </div>
                </nav>

                {/* Floating Icons */}
                <div className="absolute top-20 right-10 text-5xl animate-float opacity-70">
                    🍰
                </div>

                {/* Hero Content */}
                <div className="text-center max-w-4xl mx-auto -mt-12">
                    <h1 className="font-serif text-5xl sm:text-6xl md:text-7xl font-bold text-slate-900 mb-6 leading-tight">
                        Malaysian Kuih
                        <br />
                        <span className="gradient-text">Recognition AI</span>
                    </h1>

                    <p className="text-slate-600 text-lg sm:text-xl mb-3 max-w-2xl mx-auto">
                        Identify traditional Malaysian kuih instantly with AI-powered CNN technology.
                    </p>

                    <p className="text-purple-600 text-base sm:text-lg font-medium mb-10">
                        Get <span className="font-semibold">calorie information</span> and <span className="font-semibold">cultural insights</span> powered by Gemini AI.
                    </p>

                    <button
                        onClick={scrollToApp}
                        className="inline-flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-purple-600 via-fuchsia-600 to-purple-600 hover:from-purple-700 hover:via-fuchsia-700 hover:to-purple-700 text-white font-semibold rounded-full shadow-lg hover:shadow-xl transition-all transform hover:scale-105 active:scale-95 text-lg"
                    >
                        <span>📸</span>
                        <span>Start Recognition</span>
                    </button>
                </div>

                {/* Decorative Elements */}
                <div className="absolute bottom-32 left-1/2 transform -translate-x-1/2 w-full max-w-sm">
                    <svg viewBox="0 0 400 100" className="w-full h-auto opacity-20">
                        <path
                            d="M0,50 Q100,30 200,50 T400,50 L400,100 L0,100 Z"
                            fill="url(#mountain-gradient)"
                        />
                        <defs>
                            <linearGradient id="mountain-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                                <stop offset="0%" stopColor="#a78bfa" />
                                <stop offset="100%" stopColor="#c084fc" />
                            </linearGradient>
                        </defs>
                    </svg>
                </div>

                {/* Scroll Indicator */}
                <div className="absolute bottom-8 text-center w-full px-4">
                    <button
                        onClick={scrollToApp}
                        className="flex flex-col items-center gap-2 mx-auto hover:text-purple-600 transition-colors"
                    >
                        <p className="text-slate-600 text-sm font-medium">Scroll to try it out</p>
                        <div className="w-6 h-10 border-2 border-slate-400 rounded-full flex items-start justify-center p-1">
                            <div className="w-1.5 h-2 bg-slate-400 rounded-full animate-bounce"></div>
                        </div>
                    </button>
                </div>
            </div>

            {/* App Section - Full Recognition Interface */}
            <div id="app-section" className="min-h-screen bg-gradient-to-b from-slate-50 to-white py-16">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    {/* Section Header */}
                    <div className="text-center mb-12">
                        <h2 className="font-serif text-4xl font-bold text-slate-900 mb-4">
                            Try the Recognition
                        </h2>
                        <p className="text-slate-600 text-lg max-w-2xl mx-auto">
                            Upload an image of a Malaysian kuih and let our AI identify it.
                        </p>
                    </div>

                    {/* Status Messages */}
                    <div className="mb-8">
                        {error && (
                            <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-red-700 flex items-center gap-3 max-w-4xl mx-auto">
                                <i className="fa-solid fa-circle-exclamation"></i> {error}
                            </div>
                        )}
                        {!error && (
                            <div className="flex items-center justify-center gap-2 text-sm text-emerald-600 font-medium">
                                <span className="relative flex h-3 w-3">
                                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                                    <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
                                </span>
                                System Active: CNN & Gemini AI Connected
                            </div>
                        )}
                    </div>

                    {/* Main Recognition Interface */}
                    <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 mb-16">
                        {/* Left: Upload & Available Classes */}
                        <div className="lg:col-span-4 space-y-6">
                            <ImageUpload
                                onUpload={handleImageUpload}
                                isLoading={isLoading}
                                modelLoaded={true}
                            />

                            {/* Always show available kuih classes */}
                            <KuihListing />
                        </div>

                        {/* Right: Results */}
                        <div className="lg:col-span-8 space-y-6">
                            {predictionResult && predictionResult.success ? (
                                <>
                                    {/* Prediction Result Card */}
                                    <div className="bg-white rounded-2xl shadow-lg border border-slate-200 overflow-hidden">
                                        <div className="grid md:grid-cols-2">
                                            <div className="h-64 md:h-auto relative bg-slate-100">
                                                <img
                                                    src={`http://localhost:5000/uploads/${predictionResult.image_path}`}
                                                    className="absolute inset-0 w-full h-full object-cover"
                                                    alt="Uploaded Kuih"
                                                />
                                            </div>
                                            <div className="p-6 flex flex-col justify-center">
                                                <div className="mb-1 text-xs font-bold tracking-wider text-emerald-600 uppercase">
                                                    CNN Prediction Result
                                                </div>
                                                <h3 className="font-serif text-3xl font-bold text-slate-800 mb-2">
                                                    {predictionResult.kuih_name}
                                                </h3>
                                                <div className="flex items-center gap-4 mb-4">
                                                    <div className="px-3 py-1 rounded-full bg-emerald-50 text-emerald-700 text-xs font-bold border border-emerald-100">
                                                        {predictionResult.confidence} Confidence
                                                    </div>
                                                </div>

                                                {/* Unified Calorie & Portion Card */}
                                                <div className="grid grid-cols-2 gap-4 mb-6">
                                                    {/* Weight Card */}
                                                    <div className="bg-gradient-to-br from-slate-50 to-emerald-50 rounded-xl border border-slate-200 p-4">
                                                        <div className="flex items-center justify-between mb-3">
                                                            <div className="text-xs font-bold text-slate-700 uppercase tracking-wide">
                                                                Avg. Weight
                                                            </div>
                                                            <i className="fa-solid fa-scale-balanced text-slate-400"></i>
                                                        </div>
                                                        <div className="text-2xl font-bold text-slate-800 leading-none">
                                                            {predictionResult.weight && predictionResult.weight !== 'N/A' ? predictionResult.weight : 'N/A'}
                                                            <span className="text-sm text-slate-500 font-medium ml-1">g</span>
                                                        </div>
                                                        <div className="text-xs text-slate-400 mt-1">per piece</div>
                                                    </div>

                                                    {/* Calorie Card */}
                                                    <div className="bg-gradient-to-br from-orange-50 to-amber-50 rounded-xl border border-orange-100 p-4">
                                                        <div className="flex items-center justify-between mb-3">
                                                            <div className="text-xs font-bold text-orange-700 uppercase tracking-wide">
                                                                Calories
                                                            </div>
                                                            <i className="fa-solid fa-fire text-orange-400"></i>
                                                        </div>
                                                        <div className="text-2xl font-bold text-slate-800 leading-none">
                                                            {predictionResult.calories}
                                                            <span className="text-sm text-slate-500 font-medium ml-1">kcal</span>
                                                        </div>
                                                        <div className="text-xs text-orange-400/80 mt-1">per piece</div>
                                                    </div>
                                                </div>

                                                {/* Calculation Section */}
                                                <div className="bg-white rounded-xl border border-slate-200 p-4 mb-6">
                                                    <div className="flex items-end gap-4">
                                                        <div className="flex-1">
                                                            <label className="block text-xs text-slate-500 mb-1.5 font-bold uppercase tracking-wider">
                                                                Serving Quantity
                                                            </label>
                                                            <div className="relative">
                                                                <input
                                                                    type="number"
                                                                    value={pieces}
                                                                    onChange={(e) => setPieces(Math.max(0, parseFloat(e.target.value) || 0))}
                                                                    min="0"
                                                                    step="0.5"
                                                                    className="w-full pl-3 pr-8 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm font-semibold text-slate-700 focus:ring-2 focus:ring-emerald-500 focus:border-transparent outline-none transition-shadow"
                                                                />
                                                                <div className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 text-xs pointer-events-none">
                                                                    pcs
                                                                </div>
                                                            </div>
                                                        </div>

                                                        <div className="flex-1 text-right">
                                                            <div className="text-xs text-slate-500 mb-0.5 font-bold uppercase tracking-wider">Total Intake</div>
                                                            <div className="text-3xl font-bold text-emerald-600 leading-none">
                                                                {calculateTotalCalories(predictionResult.calories)}
                                                                <span className="text-sm text-emerald-600/70 font-medium ml-1">kcal</span>
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>

                                                {/* Feedback Section */}
                                                {!feedbackSent ? (
                                                    <div className="border-t border-slate-100 pt-4 mt-auto">
                                                        <p className="text-xs text-slate-400 mb-2">Is this prediction accurate?</p>
                                                        {!showCorrection ? (
                                                            <div className="flex gap-2">
                                                                <button
                                                                    onClick={() => handleFeedback(true)}
                                                                    className="flex-1 py-1.5 rounded-lg border border-slate-200 text-xs font-medium hover:bg-emerald-50 hover:text-emerald-600 hover:border-emerald-200 transition-colors"
                                                                >
                                                                    Yes, Correct
                                                                </button>
                                                                <button
                                                                    onClick={() => handleFeedback(false)}
                                                                    className="flex-1 py-1.5 rounded-lg border border-slate-200 text-xs font-medium hover:bg-red-50 hover:text-red-600 hover:border-red-200 transition-colors"
                                                                >
                                                                    No, Incorrect
                                                                </button>
                                                            </div>
                                                        ) : (
                                                            <div className="flex gap-2 animate-fade-in-up">
                                                                <select
                                                                    className="flex-1 text-xs border border-slate-300 rounded-lg px-2 py-1.5 bg-white"
                                                                    value={actualKuih}
                                                                    onChange={(e) => setActualKuih(e.target.value)}
                                                                >
                                                                    <option value="">-- Select Actual Kuih --</option>
                                                                    {[
                                                                        'Kuih Keria', 'Kuih Ketayap', 'Kuih Lapis', 'Kuih Seri Muka',
                                                                        'Onde Onde', 'Kuih Talam', 'Kuih Cara', 'Tepung Pelita',
                                                                        'Kuih Bingka', 'Kuih Bahulu', 'Apam Balik'
                                                                    ].map(k => <option key={k} value={k}>{k}</option>)}
                                                                </select>
                                                                <button
                                                                    onClick={submitCorrection}
                                                                    className="bg-slate-800 text-white text-xs px-3 rounded-lg hover:bg-black"
                                                                >
                                                                    Submit
                                                                </button>
                                                            </div>
                                                        )}
                                                    </div>
                                                ) : (
                                                    <div className="mt-4 p-2 bg-emerald-50 text-emerald-700 text-xs text-center rounded-lg border border-emerald-100">
                                                        Thanks for your feedback!
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    </div>

                                    {/* AI Analysis Hub (Unified) */}
                                    <AIAnalysisHub
                                        kuihName={predictionResult.kuih_name}
                                        imagePath={predictionResult.image_path}
                                        calories={predictionResult.calories}
                                    />
                                </>
                            ) : (
                                <div className="h-full flex flex-col items-center justify-center text-center p-12 bg-white rounded-2xl border border-dashed border-slate-300">
                                    <div className="w-20 h-20 bg-slate-50 rounded-full flex items-center justify-center mb-4">
                                        <i className="fa-solid fa-bowl-food text-3xl text-slate-300"></i>
                                    </div>
                                    <h3 className="text-lg font-medium text-slate-600">No Analysis Yet</h3>
                                    <p className="max-w-xs mx-auto text-sm mt-2 text-slate-400">
                                        Upload an image to see CNN classification and AI insights.
                                    </p>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Feature Cards */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                        <div className="bg-white rounded-2xl p-6 shadow-md border border-slate-200 hover:shadow-xl transition-shadow">
                            <div className="text-4xl mb-4">🔍</div>
                            <h3 className="font-serif font-bold text-xl text-slate-900 mb-2">CNN Recognition</h3>
                            <p className="text-slate-600 text-sm">Advanced deep learning model trained on 11 types of Malaysian kuih</p>
                        </div>

                        <div className="bg-white rounded-2xl p-6 shadow-md border border-slate-200 hover:shadow-xl transition-shadow">
                            <div className="text-4xl mb-4">🔥</div>
                            <h3 className="font-serif font-bold text-xl text-slate-900 mb-2">Calorie Tracking</h3>
                            <p className="text-slate-600 text-sm">Get accurate calorie information and track your intake</p>
                        </div>

                        <div className="bg-white rounded-2xl p-6 shadow-md border border-slate-200 hover:shadow-xl transition-shadow">
                            <div className="text-4xl mb-4">✨</div>
                            <h3 className="font-serif font-bold text-xl text-slate-900 mb-2">Gemini AI Insights</h3>
                            <p className="text-slate-600 text-sm">Learn about cultural significance and interesting facts</p>
                        </div>
                    </div>
                </div>
            </div>

            {/* Loading Overlay */}
            {isLoading && (
                <div className="fixed inset-0 bg-slate-900/80 backdrop-blur-sm z-[60] flex flex-col items-center justify-center text-white">
                    <div className="relative w-20 h-5 mb-4">
                        <div className="absolute top-0 left-2 w-3 h-3 rounded-full bg-emerald-400 animate-pulse"></div>
                        <div className="absolute top-0 left-8 w-3 h-3 rounded-full bg-emerald-400 animate-pulse delay-150"></div>
                        <div className="absolute top-0 left-14 w-3 h-3 rounded-full bg-emerald-400 animate-pulse delay-300"></div>
                    </div>
                    <h3 className="text-xl font-serif font-bold tracking-wide">Processing Image</h3>
                    <p className="text-slate-300 text-sm mt-2">Running CNN & Vision Analysis...</p>
                </div>
            )}

            {/* History Panel */}
            <HistoryPanel isOpen={historyOpen} onClose={() => setHistoryOpen(false)} />
        </div>
    );
};

export default LandingPage;
