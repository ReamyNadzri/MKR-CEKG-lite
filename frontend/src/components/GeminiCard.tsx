import { useState, useEffect } from 'react';
import { getGeminiInfo, type GeminiInfoResponse } from '../services/api';

interface GeminiCardProps {
    kuihName: string;
}

const GeminiCard = ({ kuihName }: GeminiCardProps) => {
    const [geminiData, setGeminiData] = useState<GeminiInfoResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchGeminiInfo = async () => {
            setLoading(true);
            setError(null);

            try {
                const response = await getGeminiInfo(kuihName);
                setGeminiData(response.data);
            } catch (err: any) {
                setError(err.response?.data?.error || 'Failed to load AI insights');
                console.error('Gemini error:', err);
            } finally {
                setLoading(false);
            }
        };

        if (kuihName) {
            fetchGeminiInfo();
        }
    }, [kuihName]);

    if (loading) {
        return (
            <div className="bg-gradient-to-br from-purple-50 to-fuchsia-50 rounded-2xl p-8 border border-purple-200 animate-pulse">
                <div className="flex items-center gap-3 mb-4">
                    <div className="w-10 h-10 bg-purple-200 rounded-lg"></div>
                    <div className="h-6 bg-purple-200 rounded w-32"></div>
                </div>
                <div className="space-y-3">
                    <div className="h-4 bg-purple-200 rounded w-full"></div>
                    <div className="h-4 bg-purple-200 rounded w-3/4"></div>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="bg-red-50 rounded-2xl p-6 border border-red-200">
                <div className="flex items-center gap-2 text-red-700">
                    <i className="fa-solid fa-exclamation-triangle"></i>
                    <span className="font-medium">{error}</span>
                </div>
                <p className="text-sm text-red-600 mt-2">Gemini AI insights are not available at the moment.</p>
            </div>
        );
    }

    if (!geminiData) {
        return null;
    }

    return (
        <div className="bg-gradient-to-br from-purple-50 to-fuchsia-50 rounded-2xl p-8 border border-purple-200 shadow-lg">
            {/* Header */}
            <div className="flex items-center gap-3 mb-6">
                <div className="w-10 h-10 bg-gradient-to-br from-purple-600 to-fuchsia-600 rounded-lg flex items-center justify-center text-white">
                    ✨
                </div>
                <h3 className="font-serif font-bold text-2xl text-slate-900">AI Cultural Insights</h3>
            </div>

            {/* Description */}
            {geminiData.description && (
                <div className="mb-6">
                    <h4 className="font-semibold text-slate-700 mb-2 text-sm uppercase tracking-wide">About</h4>
                    <p className="text-slate-600 leading-relaxed">{geminiData.description}</p>
                </div>
            )}

            {/* Other Names */}
            {geminiData.othersname && (
                <div className="mb-6">
                    <h4 className="font-semibold text-slate-700 mb-2 text-sm uppercase tracking-wide">Also Known As</h4>
                    <p className="text-slate-600">{geminiData.othersname}</p>
                </div>
            )}

            {/* Estimated Calories */}
            {geminiData.estimatedcalories && (
                <div className="mb-6">
                    <h4 className="font-semibold text-slate-700 mb-2 text-sm uppercase tracking-wide">Estimated Calories</h4>
                    <p className="text-slate-600">{geminiData.estimatedcalories}</p>
                </div>
            )}

            {/* Fun Fact */}
            {geminiData.fun_fact && (
                <div className="bg-white/60 rounded-xl p-4 border border-purple-100">
                    <h4 className="font-semibold text-purple-700 mb-2 flex items-center gap-2">
                        <span>💡</span>
                        <span>Fun Fact</span>
                    </h4>
                    <p className="text-slate-600 text-sm leading-relaxed">{geminiData.fun_fact}</p>
                </div>
            )}

            {/* AI Badge */}
            <div className="mt-6 pt-4 border-t border-purple-200 flex items-center justify-between">
                <span className="text-xs text-slate-500">Powered by Gemini AI</span>
                <span className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-xs font-medium">AI Generated</span>
            </div>
        </div>
    );
};

export default GeminiCard;
