export interface PredictionResponse {
  prediction: string;
  confidence: number;
}

export interface UploadFormData {
  image: FileList;
}

export interface ApiError {
  message: string;
}

export const TUMOR_CLASSES = [
  'Astrocitoma',
  'Ependimoma',
  'Glioma',
  'Hemangiopericitoma',
  'Meningioma',
  'Neurocitoma',
  'Oligodendroglioma',
  'Schwannoma',
  'Outros Tumores',
  'Normal (Sem Tumor)',
] as const;

export type TumorClass = (typeof TUMOR_CLASSES)[number];

export const ACCEPTED_IMAGE_TYPES = [
  'image/png',
  'image/jpeg',
  'image/jpg',
] as const;

export const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024;
