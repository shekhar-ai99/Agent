import axios from 'axios'

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api'

const client = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  }
})

// Interceptor for error handling
client.interceptors.response.use(
  response => response,
  error => {
    console.error('API Error:', error)
    return Promise.reject(error)
  }
)

const apiClient = {
  // Run endpoints
  runBacktest: (config) => client.post('/run/backtest', config),
  runSimulation: (config) => client.post('/run/simulation', config),
  
  // Results endpoints
  getResults: (runId) => client.get(`/results/${runId}`),
  getTrades: (runId) => client.get(`/results/${runId}/trades`),
  getStrategyRanking: () => client.get('/results/strategy_ranking.json'),
  
  // Status checks
  checkRunStatus: (runId) => client.get(`/results/${runId}/status`),
  
  // Generic GET / POST helpers
  get: (endpoint) => client.get(endpoint),
  post: (endpoint, data) => client.post(endpoint, data),
}

export default apiClient
