import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import './App.css'
import { NavBar } from './components/NavBar'
import { CorePage } from './pages/CorePage'
import { OverviewPage } from './pages/OverviewPage'

function App() {
  return (
    <BrowserRouter>
      <NavBar />
      <main className="page-content">
        <Routes>
          <Route path="/" element={<Navigate to="/overview" replace />} />
          <Route path="/overview" element={<OverviewPage />} />
          <Route path="/core" element={<CorePage />} />
          <Route path="/multigroup" element={<Navigate to="/core" replace />} />
          <Route path="*" element={<Navigate to="/overview" replace />} />
        </Routes>
      </main>
    </BrowserRouter>
  )
}

export default App
