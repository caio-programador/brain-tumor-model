import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FaStethoscope } from 'react-icons/fa';
import Header from '../../components/Header';
import Loading from '../../components/Loading';
import PageContainer from '../../components/PageContainer';
import ResultCard from '../../components/ResultCard';
import UploadForm from '../../components/UploadForm';
import { predictTumor } from '../../services/api';
import type { PredictionResponse } from '../../types/api';
import styles from './Home.module.css';

function HomePage(): React.JSX.Element {
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);

  const handleAnalyze = async (file: File): Promise<void> => {
    setIsLoading(true);
    setResult(null);

    try {
      const prediction = await predictTumor(file);
      setResult(prediction);
    } catch {
      void navigate('/error');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
      <Header />
      <PageContainer>
        <section className={styles.hero}>
          <h1 className={styles.title}>
            Classificação de Tumores Cerebrais por Inteligência Artificial
          </h1>
          <p className={styles.subtitle}>
            Envie uma imagem de ressonância magnética T1C+ para análise automática
            utilizando um modelo de Deep Learning baseado em ResNet18.
          </p>
        </section>

        <section className={styles.mainCard} aria-label="Formulário de upload">
          <h2 className={styles.cardTitle}>
            <FaStethoscope className={styles.cardTitleIcon} aria-hidden="true" />
            Análise de Imagem
          </h2>

          {isLoading ? (
            <Loading />
          ) : (
            <UploadForm
              onSubmit={(file) => {
                void handleAnalyze(file);
              }}
              isLoading={isLoading}
            />
          )}
        </section>

        {result && !isLoading && (
          <section className={styles.resultSection} aria-label="Resultado da análise">
            <ResultCard result={result} />
          </section>
        )}
      </PageContainer>
    </>
  );
}

export default HomePage;
