import { Navigate, Route, Routes } from 'react-router-dom';
import ErrorPage from '../pages/Error';
import HomePage from '../pages/Home';

function AppRoutes(): React.JSX.Element {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/error" element={<ErrorPage />} />
      <Route path="*" element={<Navigate to="/error" replace />} />
    </Routes>
  );
}

export default AppRoutes;
