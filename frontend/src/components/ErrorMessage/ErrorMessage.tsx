import { MdErrorOutline } from 'react-icons/md';
import styles from './ErrorMessage.module.css';

interface ErrorMessageProps {
  message: string;
}

function ErrorMessage({ message }: ErrorMessageProps): React.JSX.Element {
  return (
    <div className={styles.errorMessage} role="alert">
      <MdErrorOutline className={styles.errorIcon} aria-hidden="true" />
      <span>{message}</span>
    </div>
  );
}

export default ErrorMessage;
