
import GeminiCard from './GeminiCard';
import PosterGenerator from './PosterGenerator';

interface AIAnalysisHubProps {
    kuihName: string;
    imagePath: string;
    calories: string;
}

const AIAnalysisHub = ({ kuihName, imagePath, calories }: AIAnalysisHubProps) => {
    return (
        <div className="space-y-6">
            {/* Cultural Insights */}
            <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
                <div className="px-6 py-4 border-b border-slate-100 bg-slate-50/50 flex items-center gap-2">
                    <i className="fa-solid fa-sparkles text-purple-600"></i>
                    <h3 className="font-bold text-slate-700">Cultural Insights</h3>
                </div>
                <div className="p-6">
                    <GeminiCard kuihName={kuihName} />
                </div>
            </div>

            {/* Poster Generator */}
            <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
                <div className="px-6 py-4 border-b border-slate-100 bg-slate-50/50 flex items-center gap-2">
                    <i className="fa-solid fa-palette text-purple-600"></i>
                    <h3 className="font-bold text-slate-700">Recipe Poster</h3>
                </div>
                <div className="p-6">
                    <PosterGenerator
                        kuihName={kuihName}
                        imagePath={imagePath}
                        calories={calories}
                    />
                </div>
            </div>
        </div>
    );
};

export default AIAnalysisHub;
