import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Layout from './components/common/Layout'
import RunConfig from './pages/RunConfig'
import RunResults from './pages/RunResults'
import TradeDetails from './pages/TradeDetails'
import StrategyRanking from './pages/StrategyRanking'

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<RunConfig />} />
          <Route path="/results/:runId" element={<RunResults />} />
          <Route path="/trades/:runId" element={<TradeDetails />} />
          <Route path="/rankings" element={<StrategyRanking />} />
        </Routes>
      </Layout>
    </Router>
  )
}

export default App
