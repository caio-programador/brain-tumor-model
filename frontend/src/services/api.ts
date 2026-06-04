import axios, { type AxiosError } from 'axios';
import type { ApiError, PredictionResponse } from '../types/api';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL as string,
});

export async function predictTumor(
  file: File,
): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append('image', file);

  const response = await api.post<PredictionResponse>('/predict', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
}

export function getApiErrorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError<ApiError>;
    return axiosError.response?.data?.message ?? axiosError.message;
  }

  if (error instanceof Error) {
    return error.message;
  }

  return 'Erro desconhecido ao processar a requisição.';
}

export default api;
