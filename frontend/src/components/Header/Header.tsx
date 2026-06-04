import { FaHospital } from 'react-icons/fa';
import styles from './Header.module.css';

function Header(): React.JSX.Element {
  return (
    <header className={styles.header}>
      <div className={styles.headerInner}>
        <div className={styles.logoWrapper}>
          <FaHospital className={styles.logoIcon} aria-hidden="true" />
        </div>
        <div className={styles.brandText}>
          <span className={styles.appName}>Brain Tumor AI</span>
          <span className={styles.appTagline}>
            Diagnóstico assistido por Inteligência Artificial
          </span>
        </div>
      </div>
    </header>
  );
}

export default Header;
