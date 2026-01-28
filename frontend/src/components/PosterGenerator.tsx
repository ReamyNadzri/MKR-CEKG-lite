import { useState, useEffect } from 'react';
import {
    generatePoster,
    getQuotaStatus,
    unlockPoster,
    type QuotaResponse
} from '../services/api';

interface PosterGeneratorProps {
    kuihName: string;
    imagePath: string;
    calories: string;
}

const PosterGenerator = ({ kuihName, imagePath, calories }: PosterGeneratorProps) => {
    const [quota, setQuota] = useState<QuotaResponse | null>(null);
    const [unlockCode, setUnlockCode] = useState('');
    const [generating, setGenerating] = useState(false);
    // const [jobId, setJobId] = useState<string | null>(null); // Removed polling
    const [posterImage, setPosterImage] = useState<string | null>(null);

    const [error, setError] = useState<string | null>(null);
    const [showUnlock, setShowUnlock] = useState(false);

    useEffect(() => {
        fetchQuota();
    }, []);

    // Polling effect removed
    // useEffect(() => {
    //     if (jobId) {
    //         const interval = setInterval(checkPosterStatus, 2000);
    //         return () => clearInterval(interval);
    //     }
    // }, [jobId]);

    const fetchQuota = async () => {
        try {
            const response = await getQuotaStatus();
            setQuota(response.data);
        } catch (err) {
            console.error('Failed to fetch quota:', err);
        }
    };

    const handleUnlock = async () => {
        if (!unlockCode.trim()) {
            setError('Please enter an unlock code');
            return;
        }

        try {
            const response = await unlockPoster(unlockCode);
            if (response.data.success) {
                await fetchQuota();
                setUnlockCode('');
                setError(null);
                alert('Unlock successful! You now have unlimited poster generation.');
            } else {
                setError(response.data.error || 'Invalid code');
            }
        } catch (err: any) {
            setError(err.response?.data?.error || 'Failed to unlock');
        }
    };

    const handleGenerate = async () => {
        if (!quota || (quota.remaining <= 0 && !quota.unlocked)) {
            setError('Quota exceeded. Please wait or unlock unlimited access.');
            return;
        }

        setGenerating(true);
        setError(null);
        setPosterImage(null);

        try {
            const response = await generatePoster({
                kuih: kuihName,
                image_filename: imagePath,
                calories: calories
            });

            if (response.data.image_base64) {
                // Handle synchronous response directly
                // Remove "data:image/png;base64," prefix if it exists to avoid double prefixing provided by backend
                const cleanBase64 = response.data.image_base64.replace(/^data:image\/\w+;base64,/, '');
                setPosterImage(cleanBase64);
                await fetchQuota();
            } else if (response.data.job_id) {
                // Keep legacy support for async if needed, but for now we expect sync
                // setJobId(response.data.job_id); 
            } else {
                setError(response.data.error || 'Failed to generate poster');
            }
        } catch (err: any) {
            setError(err.response?.data?.error || 'Failed to generate poster');
        } finally {
            setGenerating(false);
        }
    };

    // checkPosterStatus removed as not needed for sync response

    const downloadPoster = () => {
        if (!posterImage) return;

        const link = document.createElement('a');
        link.href = `data:image/png;base64,${posterImage}`;
        link.download = `${kuihName}_poster.png`;
        link.click();
    };

    const handleShare = async () => {
        if (!posterImage) return;

        try {
            // Convert base64 to blob
            const byteCharacters = atob(posterImage);
            const byteNumbers = new Array(byteCharacters.length);
            for (let i = 0; i < byteCharacters.length; i++) {
                byteNumbers[i] = byteCharacters.charCodeAt(i);
            }
            const byteArray = new Uint8Array(byteNumbers);
            const blob = new Blob([byteArray], { type: 'image/png' });
            const file = new File([blob], `${kuihName}_poster.png`, { type: 'image/png' });

            if (navigator.share) {
                await navigator.share({
                    title: `My ${kuihName} Poster`,
                    text: `Check out this AI-generated poster for ${kuihName}!`,
                    files: [file]
                });
            } else {
                alert('Sharing is not supported on this device/browser. Please download the image instead.');
            }
        } catch (error) {
            console.error('Error sharing:', error);
        }
    };

    return (
        <div className="bg-gradient-to-br from-purple-50 to-fuchsia-50 rounded-2xl p-6 border border-purple-200">
            <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 bg-gradient-to-br from-purple-600 to-fuchsia-600 rounded-lg flex items-center justify-center text-white">
                    🎨
                </div>
                <div>
                    <div className="flex items-center gap-3">
                        <h3 className="font-serif font-bold text-xl text-slate-900">AI Recipe Poster</h3>
                        <span className="px-2 py-0.5 rounded-full bg-amber-100 text-amber-700 text-[10px] font-bold uppercase tracking-wider border border-amber-200">
                            Experimental
                        </span>
                    </div>
                    {/* <p className="text-xs text-slate-500">Create beautiful posters with Gemini AI</p> */}
                </div>
            </div>

            {/* Disclaimer */}
            <div className="mb-4 p-3 bg-slate-50 border border-slate-200 rounded-lg flex gap-2">
                <i className="fa-solid fa-circle-info text-slate-400 mt-0.5 text-xs"></i>
                <p className="text-xs text-slate-500 leading-relaxed">
                    <span className="font-bold text-slate-600">Disclaimer:</span> AI-generated posters are experimental and may contain inaccurate information, ingredient lists, or cooking steps. Please verify all details before use.
                </p>
            </div>

            {/* Quota Display */}
            {quota && (
                <>
                    <div className="mb-4 p-3 bg-white/60 rounded-xl border border-purple-100">
                        <div className="flex justify-between items-center">
                            <span className="text-sm font-bold text-slate-700 flex items-center gap-2">
                                <i className="fa-solid fa-palette text-purple-600"></i>
                                Poster Generation
                            </span>
                            <span className="text-xs font-bold text-slate-600 bg-white px-2 py-1 rounded-md border border-slate-200">
                                {quota.remaining}/2 remaining
                            </span>
                            {quota.reset_time && (
                                <span className="text-xs text-slate-500">
                                    Resets: {new Date(quota.reset_time).toLocaleTimeString()}
                                </span>
                            )}
                        </div>
                    </div>
                    {/* Unlock Toggle - Only show if quota is low */}
                    {quota.remaining < 2 && (
                        <button
                            onClick={() => setShowUnlock(!showUnlock)}
                            className="text-[10px] font-bold text-purple-600 hover:text-purple-700 flex items-center gap-1.5 mt-2 transition-colors uppercase tracking-wide"
                        >
                            <i className={`fa-solid ${showUnlock ? 'fa-chevron-up' : 'fa-lock'}`}></i>
                            {showUnlock ? 'Hide Refill Option' : 'Unlock Additional Quota'}
                        </button>
                    )}
                </>
            )}


            {/* Unlock Code Section */}
            {
                quota && !quota.unlocked && showUnlock && (
                    <div className="mb-4 animate-fade-in-up">
                        <label className="block text-sm font-medium text-slate-700 mb-2">
                            Have an unlock code?
                        </label>
                        <div className="flex gap-2">
                            <input
                                type="text"
                                value={unlockCode}
                                onChange={(e) => setUnlockCode(e.target.value.toUpperCase())}
                                placeholder="Enter unlock code"
                                className="flex-1 px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                            />
                            <button
                                onClick={handleUnlock}
                                className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg text-sm font-medium transition-colors"
                            >
                                Unlock
                            </button>
                        </div>
                    </div>
                )
            }

            {/* Error Message */}
            {
                error && (
                    <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
                        <i className="fa-solid fa-exclamation-triangle mr-2"></i>
                        {error}
                    </div>
                )
            }

            {/* Quota Exhausted Warning */}
            {quota && quota.remaining <= 0 && !quota.unlocked && (
                <div className="mb-4 p-3 bg-amber-50 border border-amber-200 rounded-lg text-amber-800 text-sm flex gap-2 items-start">
                    <i className="fa-solid fa-clock mt-0.5"></i>
                    <div>
                        <p className="font-bold">Daily generation limit reached</p>
                        <p className="text-xs mt-0.5 opacity-90">
                            You've used your free poster generations.
                            {quota.reset_time ? ` Resets at ${new Date(quota.reset_time).toLocaleTimeString()}` : ' Resets in 3 hours'}.
                        </p>
                    </div>
                </div>
            )}

            {/* Generate Button */}
            <button
                onClick={handleGenerate}
                disabled={generating || !!(quota && quota.remaining <= 0 && !quota.unlocked)}
                className="w-full py-3 px-4 bg-gradient-to-r from-purple-600 to-fuchsia-600 hover:from-purple-700 hover:to-fuchsia-700 text-white rounded-xl font-semibold shadow-lg transition-all flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
                {generating ? (
                    <>
                        <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                        <span>Generating Poster...</span>
                    </>
                ) : (
                    <>
                        <i className="fa-solid fa-wand-magic-sparkles"></i>
                        <span className="font-bold">Generate AI Recipe Poster</span>
                    </>
                )}
            </button>

            {/* Poster Display */}
            {
                posterImage && (
                    <div className="mt-6">
                        <img
                            src={`data:image/png;base64,${posterImage}`}
                            alt="Generated Poster"
                            className="w-full rounded-xl shadow-lg border-2 border-white mb-4"
                        />
                        <div className="flex gap-2">
                            <button
                                onClick={downloadPoster}
                                className="flex-1 py-2.5 px-4 bg-white hover:bg-gray-50 text-purple-600 rounded-xl font-semibold border-2 border-purple-200 transition-colors flex items-center justify-center gap-2"
                            >
                                <i className="fa-solid fa-download"></i>
                                Download
                            </button>
                            <button
                                onClick={handleShare}
                                className="flex-1 py-2.5 px-4 bg-purple-100 hover:bg-purple-200 text-purple-700 rounded-xl font-semibold border-2 border-purple-200 transition-colors flex items-center justify-center gap-2"
                            >
                                <i className="fa-solid fa-share-nodes"></i>
                                Share
                            </button>
                        </div>
                    </div>
                )
            }
        </div >
    );
};

export default PosterGenerator;
