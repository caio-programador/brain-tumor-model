import { FaBrain, FaCheckCircle, FaExclamationTriangle } from 'react-icons/fa';
import type { PredictionResponse } from '../../types/api';
import styles from './ResultCard.module.css';

interface ResultCardProps {
  result: PredictionResponse;
}

function isNormalPrediction(prediction: string): boolean {
  return prediction.toLowerCase().includes('normal');
}

function ResultCard({ result }: ResultCardProps): React.JSX.Element {
  const isNormal = isNormalPrediction(result.prediction);
  const confidencePercent = result.confidence.toFixed(2);

  return (
    <article className={styles.card} aria-label="Resultado da análise">
      <div className={styles.cardHeader}>
        <FaBrain className={styles.headerIcon} aria-hidden="true" />
        <h2 className={styles.headerTitle}>Resultado da IA</h2>
      </div>

      <div className={styles.cardBody}>
        <div className={styles.predictionWrapper}>
          <span className={styles.predictionLabel}>Classificação</span>
          <p className={styles.predictionValue}>{result.prediction}</p>
          <span
            className={`${styles.badge} ${isNormal ? styles.badgeNormal : styles.badgeTumor}`}
          >
            {isNormal ? (
              <>
                <FaCheckCircle aria-hidden="true" />
                Sem tumor detectado
              </>
            ) : (
              <>
                <FaExclamationTriangle aria-hidden="true" />
                Tumor identificado
              </>
            )}
          </span>
        </div>

        <div className={styles.confidenceSection}>
          <span className={styles.confidenceLabel}>Confiança:</span>
          <div className={styles.confidenceBarWrapper}>
            <div className={styles.confidenceBar} role="progressbar" aria-valuenow={result.confidence} aria-valuemin={0} aria-valuemax={100}>
              <div
                className={styles.confidenceFill}
                style={{ width: `${Math.min(result.confidence, 100)}%` }}
              />
            </div>
            <span className={styles.confidenceValue}>{confidencePercent}%</span>
          </div>
        </div>
      </div>
    </article>
  );
}

export default ResultCard;
