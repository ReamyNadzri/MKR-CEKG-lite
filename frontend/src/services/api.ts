import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export type PredictionResponse = {
    success: boolean;
    kuih_name: string;
    confidence: string;
    calories: string;
    weight?: string; // Added weight field
    image_path: string;
    is_gemini_prediction: boolean;
    available_classes: string[];
};

export type GeminiInfoResponse = {
    description: string;
    othersname: string;
    estimatedcalories: string;
    fun_fact: string;
    error?: string;
};

export type PosterJobResponse = {
    job_id?: string;
    image_base64?: string; // Added for synchronous response
    quota_exceeded?: boolean;
    remaining?: number;
    error?: string;
};

export type PosterStatusResponse = {
    status: 'PENDING' | 'PROCESSING' | 'COMPLETED' | 'FAILED';
    result?: {
        image_base64: string;
    };
    error?: string;
};

export type QuotaResponse = {
    remaining: number;
    unlocked: boolean;
    reset_time?: string;
};

export type HistoryResponse = {
    _id?: string;
    kuih_name: string;
    calories: string;
    timestamp: string;
};

// API Functions
export const uploadImage = (formData: FormData) =>
    api.post<PredictionResponse>('/api/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
    });

export const getGeminiInfo = (kuihName: string) =>
    api.post<GeminiInfoResponse>('/gemini-info', { kuih: kuihName });

export const generatePoster = (data: {
    kuih: string;
    image_filename: string;
    calories: string;
}) => api.post<PosterJobResponse>('/generate_poster', data);

export const getPosterStatus = (jobId: string) =>
    api.get<PosterStatusResponse>(`/poster_status/${jobId}`);

export const getQuotaStatus = () => api.get<QuotaResponse>('/poster_quota');

export const unlockPoster = (code: string) =>
    api.post<{ success: boolean; message: string; error?: string }>('/unlock_poster', { code });

export const getHistory = () => api.get<HistoryResponse[]>('/api/history');

export const clearHistory = () => api.delete<{ success: boolean }>('/api/history');

export const deleteHistoryItem = (id: string) =>
    api.delete<{ success: boolean }>(`/api/history/${id}`);

export const submitFeedback = (data: {
    predicted_label: string;
    image_path: string;
    is_correct: boolean;
    actual_label?: string;
}) => api.post<{ success: boolean; message: string }>('/submit_feedback', data);

export default api;
