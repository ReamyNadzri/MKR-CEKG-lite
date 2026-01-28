import { useState, useEffect } from 'react';
import { getHistory, clearHistory, type HistoryResponse } from '../services/api';

interface HistoryPanelProps {
    isOpen: boolean;
    onClose: () => void;
}

const HistoryPanel = ({ isOpen, onClose }: HistoryPanelProps) => {
    const [history, setHistory] = useState<HistoryResponse[]>([]);
    const [loading, setLoading] = useState(false);
    const [totalCalories, setTotalCalories] = useState(0);

    useEffect(() => {
        if (isOpen) {
            fetchHistory();
        }
    }, [isOpen]);

    const fetchHistory = async () => {
        setLoading(true);
        try {
            const response = await getHistory();
            setHistory(response.data);

            // Calculate total calories
            const total = response.data.reduce((sum, item) => {
                const cal = parseInt(item.calories) || 0;
                return sum + cal;
            }, 0);
            setTotalCalories(total);
        } catch (err) {
            console.error('Failed to fetch history:', err);
        } finally {
            setLoading(false);
        }
    };

    const handleClearHistory = async () => {
        if (!confirm('Are you sure you want to clear all history?')) return;

        try {
            await clearHistory();
            setHistory([]);
            setTotalCalories(0);
        } catch (err) {
            console.error('Failed to clear history:', err);
            alert('Failed to clear history');
        }
    };

    return (
        <>
            {/* Backdrop */}
            {isOpen && (
                <div
                    className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40 transition-opacity"
                    onClick={onClose}
                />
            )}

            {/* Slide-in Panel */}
            <div className={`fixed top-0 right-0 h-full w-full sm:w-96 bg-white shadow-2xl z-50 transform transition-transform duration-300 ${isOpen ? 'translate-x-0' : 'translate-x-full'
                }`}>
                {/* Header */}
                <div className="bg-gradient-to-r from-emerald-600 to-teal-600 text-white p-6">
                    <div className="flex justify-between items-center mb-2">
                        <h2 className="text-2xl font-serif font-bold">History</h2>
                        <button
                            onClick={onClose}
                            className="text-white hover:bg-white/20 rounded-lg p-2 transition-colors"
                        >
                            <i className="fa-solid fa-times text-xl"></i>
                        </button>
                    </div>
                    <p className="text-emerald-100 text-sm">Your recent predictions</p>
                </div>

                {/* Content */}
                <div className="h-[calc(100%-12rem)] overflow-y-auto p-6">
                    {loading ? (
                        <div className="flex items-center justify-center h-full">
                            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-emerald-600"></div>
                        </div>
                    ) : history.length === 0 ? (
                        <div className="flex flex-col items-center justify-center h-full text-center">
                            <div className="w-20 h-20 bg-slate-100 rounded-full flex items-center justify-center mb-4">
                                <i className="fa-solid fa-history text-3xl text-slate-300"></i>
                            </div>
                            <h3 className="text-lg font-medium text-slate-600 mb-2">No History Yet</h3>
                            <p className="text-sm text-slate-400">Your predictions will appear here</p>
                        </div>
                    ) : (
                        <div className="space-y-4">
                            {history.map((item, index) => (
                                <div
                                    key={index}
                                    className="bg-slate-50 rounded-xl p-4 border border-slate-200 hover:shadow-md transition-shadow"
                                >
                                    <div className="flex justify-between items-start mb-2">
                                        <h4 className="font-semibold text-slate-800 text-sm">{item.kuih_name}</h4>
                                        <span className="text-xs text-slate-500">
                                            {new Date(item.timestamp).toLocaleDateString()}
                                        </span>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <span className="px-2 py-1 bg-orange-100 text-orange-700 text-xs rounded-full font-medium">
                                            {item.calories} kcal
                                        </span>
                                        <span className="text-xs text-slate-400">
                                            {new Date(item.timestamp).toLocaleTimeString()}
                                        </span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className="absolute bottom-0 left-0 right-0 bg-white border-t border-slate-200 p-6">
                    <div className="bg-gradient-to-r from-orange-50 to-red-50 rounded-xl p-4 mb-4 border border-orange-200">
                        <div className="text-xs text-orange-600 mb-1">Total Calories</div>
                        <div className="text-2xl font-bold text-slate-800">
                            {totalCalories} <span className="text-sm font-normal text-slate-500">kcal</span>
                        </div>
                    </div>

                    {history.length > 0 && (
                        <button
                            onClick={handleClearHistory}
                            className="w-full py-2.5 px-4 bg-red-50 hover:bg-red-100 text-red-600 rounded-xl font-semibold transition-colors border border-red-200"
                        >
                            <i className="fa-solid fa-trash mr-2"></i>
                            Clear All History
                        </button>
                    )}
                </div>
            </div>
        </>
    );
};

export default HistoryPanel;
