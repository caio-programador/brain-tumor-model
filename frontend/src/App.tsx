import { BrowserRouter } from 'react-router-dom';
import AppRoutes from './routes';

function App(): React.JSX.Element {
  return (
    <BrowserRouter>
      <AppRoutes />
    </BrowserRouter>
  );
}

export default App;
