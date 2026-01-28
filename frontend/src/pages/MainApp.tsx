import { useState } from 'react';
import Navigation from '../components/Navigation';
import ImageUpload from '../components/ImageUpload';
import { uploadImage, type PredictionResponse } from '../services/api';
import AIAnalysisHub from '../components/AIAnalysisHub';

const MainApp = () => {
    // UI Update: Force HMR Refresh
    const [isLoading, setIsLoading] = useState(false);
    const [modelLoaded] = useState(true); // Assume model is loaded
    const [error, setError] = useState<string | null>(null);
    const [predictionResult, setPredictionResult] = useState<PredictionResponse | null>(null);
    const [historyOpen, setHistoryOpen] = useState(false);
    const [portionCount, setPortionCount] = useState(1);

    const handleImageUpload = async (file: File) => {
        setIsLoading(true);
        setError(null);
        setPortionCount(1); // Reset portion count on new upload

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await uploadImage(formData);
            setPredictionResult(response.data);
        } catch (err: any) {
            setError(err.response?.data?.error || 'Failed to analyze image. Please try again.');
            console.error('Upload error:', err);
        } finally {
            setIsLoading(false);
        }
    };

    const toggleHistory = () => {
        setHistoryOpen(!historyOpen);
    };

    return (
        <div className="min-h-screen flex flex-col bg-slate-50">
            <Navigation onHistoryToggle={toggleHistory} />

            <main className="flex-grow max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
                {/* Status Messages */}
                <div className="mb-8">
                    {error && (
                        <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-red-700 flex items-center gap-3 animate-pulse">
                            <i className="fa-solid fa-circle-exclamation"></i> {error}
                        </div>
                    )}
                    {!error && modelLoaded && (
                        <div className="flex items-center gap-2 text-sm text-emerald-600 font-medium px-2">
                            <span className="relative flex h-3 w-3">
                                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                                <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
                            </span>
                            System Active: CNN & Gemini AI Connected
                        </div>
                    )}
                </div>

                {/* Main Grid */}
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 mb-16">
                    {/* Left Sidebar - Upload */}
                    <div className="lg:col-span-4 space-y-6">
                        <ImageUpload
                            onUpload={handleImageUpload}
                            isLoading={isLoading}
                            modelLoaded={modelLoaded}
                        />

                        {/* Available Classes */}
                        {predictionResult && predictionResult.available_classes && (
                            <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
                                <h3 className="font-semibold text-slate-700 mb-4 text-sm uppercase tracking-wider">
                                    Identifiable Kuih
                                </h3>
                                <div className="flex flex-wrap gap-2 max-h-40 overflow-y-auto">
                                    {predictionResult.available_classes.map((kuih, idx) => (
                                        <span
                                            key={idx}
                                            className="px-3 py-1 bg-slate-100 text-slate-600 text-xs rounded-full border border-slate-200"
                                        >
                                            {kuih}
                                        </span>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Right Content Area */}
                    <div className="lg:col-span-8">
                        {predictionResult && predictionResult.success ? (
                            <div className="space-y-6 animate-fade-in-up">
                                {/* Results Display */}
                                <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
                                    <div className="grid md:grid-cols-2">
                                        <div className="h-64 md:h-auto relative bg-slate-100">
                                            <img
                                                src={`http://localhost:5000/uploads/${predictionResult.image_path}`}
                                                className="absolute inset-0 w-full h-full object-cover"
                                                alt="Uploaded Kuih"
                                            />
                                            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent p-4">
                                                <p className="text-white text-xs font-medium">Input Image</p>
                                            </div>
                                        </div>
                                        <div className="p-6 flex flex-col justify-center">
                                            <div className="mb-1 text-xs font-bold tracking-wider text-emerald-600 uppercase">
                                                CNN Prediction Result
                                            </div>
                                            <h2 className="font-serif text-3xl font-bold text-slate-800 mb-2">
                                                {predictionResult.kuih_name}
                                            </h2>
                                            <div className="flex items-center gap-4 mb-6">
                                                <div className="px-3 py-1 rounded-full bg-emerald-50 text-emerald-700 text-xs font-bold border border-emerald-100">
                                                    {predictionResult.confidence} Confidence
                                                </div>
                                                {predictionResult.weight && predictionResult.weight !== 'N/A' && (
                                                    <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-slate-100 text-slate-600 text-xs font-bold border border-slate-200">
                                                        <i className="fa-solid fa-scale-balanced text-slate-400"></i>
                                                        <span>Avg {predictionResult.weight}g</span>
                                                    </div>
                                                )}
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
                                                                value={portionCount}
                                                                onChange={(e) => setPortionCount(Math.max(0, parseFloat(e.target.value) || 0))}
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
                                                            {predictionResult.calories !== 'N/A'
                                                                ? Math.round(parseFloat(predictionResult.calories) * portionCount)
                                                                : 'N/A'}
                                                            <span className="text-sm text-emerald-600/70 font-medium ml-1">kcal</span>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>

                                        </div>
                                    </div>
                                </div>

                                {/* AI Analysis Hub (Unified) */}
                                <div className="animate-fade-in-up delay-200">
                                    <AIAnalysisHub
                                        kuihName={predictionResult.kuih_name}
                                        imagePath={predictionResult.image_path}
                                        calories={predictionResult.calories}
                                    />
                                </div>
                            </div>
                        ) : (
                            <div className="h-full flex flex-col items-center justify-center text-center p-12 bg-white rounded-2xl border border-dashed border-slate-300 text-slate-400">
                                <div className="w-20 h-20 bg-slate-50 rounded-full flex items-center justify-center mb-4">
                                    <i className="fa-solid fa-bowl-food text-3xl opacity-20"></i>
                                </div>
                                <h3 className="text-lg font-medium text-slate-600">No Analysis Yet</h3>
                                <p className="max-w-xs mx-auto text-sm mt-2">
                                    Upload an image from the left panel to see CNN classification and Gemini AI
                                    insights here.
                                </p>
                            </div>
                        )}
                    </div>
                </div>
            </main>

            {/* Footer */}
            <footer className="mt-auto border-t border-slate-200 bg-white py-8">
                <div className="max-w-7xl mx-auto px-4 text-center">
                    <p className="text-sm text-slate-900 font-bold font-serif mb-2">
                        Malaysian Kuih Recognition using CNN Architecture Variants Enhanced with Gemini AI for
                        Calories Estimation and Knowledge Generation
                    </p>
                    <p className="text-xs text-slate-400">
                        © 2025 Final Year Project Implementation. All rights reserved.
                    </p>
                </div>
            </footer>

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

            {/* History Panel Placeholder */}
            {historyOpen && (
                <>
                    <div
                        onClick={toggleHistory}
                        className="fixed inset-0 bg-slate-900/20 backdrop-blur-sm z-50"
                    ></div>
                    <div className="fixed top-0 right-0 h-full w-80 bg-white shadow-2xl z-50 p-6">
                        <div className="flex justify-between items-center mb-4">
                            <h3 className="font-serif font-bold text-slate-800">Scan History</h3>
                            <button
                                onClick={toggleHistory}
                                className="text-slate-400 hover:text-red-500 transition-colors"
                            >
                                <i className="fa-solid fa-xmark text-xl"></i>
                            </button>
                        </div>
                        <p className="text-sm text-slate-400">History feature - Coming soon...</p>
                    </div>
                </>
            )}
        </div>
    );
};

export default MainApp;
