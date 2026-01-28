import { useState, type ChangeEvent, type FormEvent, type DragEvent } from 'react';

interface ImageUploadProps {
    onUpload: (file: File) => void;
    isLoading: boolean;
    modelLoaded: boolean;
}

const ImageUpload = ({ onUpload, isLoading, modelLoaded }: ImageUploadProps) => {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [isDragging, setIsDragging] = useState(false);

    const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setSelectedFile(e.target.files[0]);
        }
    };

    const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(true);
    };

    const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);
    };

    const handleDrop = (e: DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);

        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            setSelectedFile(e.dataTransfer.files[0]);
        }
    };

    const handleSubmit = (e: FormEvent) => {
        e.preventDefault();
        if (selectedFile) {
            onUpload(selectedFile);
        }
    };

    return (
        <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
            <h2 className="font-serif text-xl font-bold text-slate-800 mb-1">Image Analysis</h2>
            <p className="text-sm text-slate-500 mb-6">Upload a photo of any traditional Kuih.</p>

            <form onSubmit={handleSubmit}>
                <div
                    onClick={() => document.getElementById('fileInput')?.click()}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    className={`relative group cursor-pointer flex flex-col items-center justify-center w-full h-64 rounded-xl border-2 border-dashed transition-all duration-300 ${isDragging
                        ? 'border-emerald-400 bg-emerald-50'
                        : 'border-slate-300 bg-slate-50 hover:bg-emerald-50 hover:border-emerald-400'
                        }`}
                >
                    <div className="flex flex-col items-center justify-center pt-5 pb-6 text-center px-4">
                        <div className="w-16 h-16 mb-4 rounded-full bg-white shadow-sm flex items-center justify-center text-3xl group-hover:scale-110 transition-transform">
                            📸
                        </div>
                        <p className="mb-2 text-sm text-slate-600 font-medium">
                            {selectedFile ? selectedFile.name : 'Click or drag image here'}
                        </p>
                        <p className="text-xs text-slate-400">PNG, JPG, WEBP (Max 16MB)</p>
                        {selectedFile && (
                            <div className="mt-4 px-3 py-1 bg-emerald-100 text-emerald-700 text-xs rounded-full truncate max-w-[200px]">
                                Selected: {selectedFile.name}
                            </div>
                        )}
                    </div>
                    <input
                        type="file"
                        id="fileInput"
                        accept="image/*"
                        onChange={handleFileChange}
                        className="hidden"
                    />
                </div>

                <button
                    type="submit"
                    disabled={!modelLoaded || isLoading || !selectedFile}
                    className="mt-6 w-full py-3.5 px-4 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700 text-white rounded-xl font-semibold shadow-lg shadow-emerald-200 transition-all flex items-center justify-center gap-2 transform active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    {isLoading ? (
                        <>
                            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                            <span>Analyzing...</span>
                        </>
                    ) : (
                        <>
                            <span>Analyze Kuih</span>
                            <i className="fa-solid fa-wand-magic-sparkles"></i>
                        </>
                    )}
                </button>
            </form>
        </div>
    );
};

export default ImageUpload;
