import styles from './Loading.module.css';

interface LoadingProps {
  text?: string;
}

function Loading({ text = 'Analisando imagem...' }: LoadingProps): React.JSX.Element {
  return (
    <div className={styles.loadingWrapper} role="status" aria-live="polite">
      <div className={styles.spinner} aria-hidden="true" />
      <p className={styles.loadingText}>{text}</p>
    </div>
  );
}

export default Loading;
