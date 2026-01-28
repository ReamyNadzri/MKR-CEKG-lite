import { Link, useNavigate } from 'react-router-dom';

interface NavigationProps {
    onHistoryToggle: () => void;
}

const Navigation = ({ onHistoryToggle }: NavigationProps) => {
    const navigate = useNavigate();

    return (
        <nav className="sticky top-0 z-40 w-full glass-panel border-b border-slate-200 shadow-sm">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex justify-between items-center h-20 relative">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-emerald-800 flex items-center justify-center text-white shadow-lg text-xl">
                            🥟
                        </div>
                        <div>
                            <h1 className="font-serif font-bold text-lg leading-tight text-slate-900">
                                Malaysian Kuih Recognition
                            </h1>
                            <p className="text-[10px] text-slate-500 hidden md:block uppercase tracking-wider">
                                CNN Variants + Gemini AI
                            </p>
                        </div>
                    </div>

                    <div className="hidden md:flex absolute left-1/2 transform -translate-x-1/2 items-center gap-1 p-1 bg-slate-100/80 rounded-full border border-slate-200">
                        <Link
                            to="/overview"
                            className="px-4 py-1.5 text-xs font-semibold text-slate-600 rounded-full hover:bg-white hover:text-primary hover:shadow-sm transition-all"
                        >
                            Overview
                        </Link>
                        <Link
                            to="/system"
                            className="px-4 py-1.5 text-xs font-semibold text-slate-600 rounded-full hover:bg-white hover:text-primary hover:shadow-sm transition-all"
                        >
                            System
                        </Link>
                        <Link
                            to="/about"
                            className="px-4 py-1.5 text-xs font-semibold text-slate-600 rounded-full hover:bg-white hover:text-primary hover:shadow-sm transition-all"
                        >
                            About
                        </Link>
                    </div>

                    <div className="flex items-center gap-3">
                        <button
                            onClick={onHistoryToggle}
                            className="group flex items-center gap-2 px-4 py-2 rounded-full bg-white border border-slate-200 hover:border-primary hover:text-primary transition-all shadow-sm"
                        >
                            <i className="fa-solid fa-clock-rotate-left text-slate-400 group-hover:text-primary"></i>
                            <span className="text-sm font-medium">History</span>
                        </button>
                        <button
                            onClick={() => navigate('/app')}
                            className="p-2 rounded-full hover:bg-slate-100 text-slate-500 transition-colors"
                            title="Reset"
                        >
                            <i className="fa-solid fa-arrows-rotate"></i>
                        </button>
                    </div>
                </div>
            </div>
        </nav>
    );
};

export default Navigation;
