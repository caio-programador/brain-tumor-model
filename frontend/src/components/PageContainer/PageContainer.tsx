import type { ReactNode } from 'react';
import styles from './PageContainer.module.css';

interface PageContainerProps {
  children: ReactNode;
}

function PageContainer({ children }: PageContainerProps): React.JSX.Element {
  return <main className={styles.container}>{children}</main>;
}

export default PageContainer;
