import { useEffect, useState } from 'react';
import { useForm } from 'react-hook-form';
import { FaCloudUploadAlt, FaRobot } from 'react-icons/fa';
import ErrorMessage from '../ErrorMessage';
import {
  ACCEPTED_IMAGE_TYPES,
  MAX_FILE_SIZE_BYTES,
  type UploadFormData,
} from '../../types/api';
import styles from './UploadForm.module.css';

interface UploadFormProps {
  onSubmit: (file: File) => void;
  isLoading: boolean;
}

const ACCEPTED_EXTENSIONS = '.png,.jpg,.jpeg';

function validateImageFile(files: FileList | undefined): string | true {
  if (!files || files.length === 0) {
    return 'Selecione uma imagem de ressonância magnética para análise.';
  }

  const file = files[0];

  const isAcceptedType = (ACCEPTED_IMAGE_TYPES as readonly string[]).includes(
    file.type,
  );

  if (!isAcceptedType) {
    return 'Formato inválido. Envie apenas imagens PNG ou JPEG.';
  }

  if (file.size > MAX_FILE_SIZE_BYTES) {
    return 'A imagem excede o limite de 10 MB. Selecione um arquivo menor.';
  }

  return true;
}

function UploadForm({ onSubmit, isLoading }: UploadFormProps): React.JSX.Element {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const {
    register,
    handleSubmit,
    watch,
    formState: { errors },
  } = useForm<UploadFormData>();

  const watchedFiles = watch('image');

  useEffect(() => {
    const files = watchedFiles;
    if (!files || files.length === 0) {
      setPreviewUrl(null);
      return;
    }

    const file = files[0];
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);

    return () => {
      URL.revokeObjectURL(objectUrl);
    };
  }, [watchedFiles]);

  const selectedFile = watchedFiles?.[0];
  const validationError = errors.image?.message;

  const handleFormSubmit = (data: UploadFormData): void => {
    const file = data.image[0];
    onSubmit(file);
  };

  const uploadAreaClassName = [
    styles.uploadArea,
    selectedFile ? styles.uploadAreaHasFile : '',
    validationError ? styles.uploadAreaHasError : '',
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <form
      className={styles.form}
      onSubmit={(event) => {
        void handleSubmit(handleFormSubmit)(event);
      }}
      noValidate
    >
      <div className={uploadAreaClassName}>
        <input
          id="image-upload"
          type="file"
          accept={ACCEPTED_EXTENSIONS}
          className={styles.fileInput}
          aria-invalid={validationError ? 'true' : 'false'}
          aria-describedby={validationError ? 'image-error' : undefined}
          disabled={isLoading}
          {...register('image', {
            validate: validateImageFile,
          })}
        />
        <FaCloudUploadAlt className={styles.uploadIcon} aria-hidden="true" />
        <span className={styles.uploadTitle}>
          Arraste ou clique para enviar a imagem
        </span>
        <span className={styles.uploadHint}>
          Formatos aceitos: PNG, JPEG · Tamanho máximo: 10 MB · Modalidade: T1C+
        </span>
        {selectedFile && (
          <span className={styles.fileName}>{selectedFile.name}</span>
        )}
      </div>

      {validationError && (
        <div id="image-error">
          <ErrorMessage message={validationError} />
        </div>
      )}

      {previewUrl && (
        <div className={styles.previewWrapper}>
          <span className={styles.previewLabel}>Pré-visualização</span>
          <img
            src={previewUrl}
            alt="Pré-visualização da imagem de ressonância magnética selecionada"
            className={styles.previewImage}
          />
        </div>
      )}

      <button
        type="submit"
        className={styles.submitButton}
        disabled={isLoading}
        aria-busy={isLoading}
      >
        <FaRobot className={styles.buttonIcon} aria-hidden="true" />
        Analisar
      </button>
    </form>
  );
}

export default UploadForm;
