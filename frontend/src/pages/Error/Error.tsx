import { Link } from 'react-router-dom';
import { FaHome } from 'react-icons/fa';
import { MdOutlineErrorOutline } from 'react-icons/md';
import Header from '../../components/Header';
import styles from './Error.module.css';

function ErrorPage(): React.JSX.Element {
  return (
    <div className={styles.wrapper}>
      <Header />
      <main className={styles.content}>
        <div className={styles.errorCard}>
          <MdOutlineErrorOutline
            className={styles.alertIcon}
            aria-hidden="true"
          />
          <h1 className={styles.errorTitle}>Ops! Algo deu errado</h1>
          <p className={styles.errorMessage}>
            Não foi possível concluir a análise da imagem. Verifique sua
            conexão ou tente novamente mais tarde.
          </p>
          <Link to="/" className={styles.homeButton}>
            <FaHome className={styles.buttonIcon} aria-hidden="true" />
            Voltar ao início
          </Link>
        </div>
      </main>
    </div>
  );
}

export default ErrorPage;
