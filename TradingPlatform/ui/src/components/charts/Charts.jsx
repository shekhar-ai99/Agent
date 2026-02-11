import { Line, Bar } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
)

export function EquityCurveChart({ data }) {
  if (!data || data.length === 0) {
    return <p className="text-gray-500">No data available</p>
  }

  const chartData = {
    labels: data.map(d => d.date),
    datasets: [
      {
        label: 'Equity',
        data: data.map(d => d.equity),
        borderColor: '#0284c7',
        backgroundColor: 'rgba(2, 132, 199, 0.1)',
        borderWidth: 2,
        fill: true,
        tension: 0.4,
      },
    ],
  }

  const options = {
    responsive: true,
    maintainAspectRatio: true,
    plugins: {
      legend: {
        position: 'top',
      },
    },
    scales: {
      y: {
        beginAtZero: false,
      },
    },
  }

  return <Line data={chartData} options={options} />
}

export function DrawdownChart({ data }) {
  if (!data || data.length === 0) {
    return <p className="text-gray-500">No data available</p>
  }

  const chartData = {
    labels: data.map(d => d.date),
    datasets: [
      {
        label: 'Drawdown %',
        data: data.map(d => d.drawdown),
        backgroundColor: 'rgba(239, 68, 68, 0.5)',
        borderColor: '#ef4444',
        borderWidth: 1,
      },
    ],
  }

  const options = {
    responsive: true,
    maintainAspectRatio: true,
    plugins: {
      legend: {
        position: 'top',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  }

  return <Bar data={chartData} options={options} />
}

export function TradesPerDayChart({ data }) {
  if (!data || data.length === 0) {
    return <p className="text-gray-500">No data available</p>
  }

  const chartData = {
    labels: data.map(d => d.day),
    datasets: [
      {
        label: 'Number of Trades',
        data: data.map(d => d.count),
        backgroundColor: '#06b6d4',
        borderColor: '#0891b2',
        borderWidth: 1,
      },
    ],
  }

  const options = {
    responsive: true,
    maintainAspectRatio: true,
    plugins: {
      legend: {
        position: 'top',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        ticks: {
          stepSize: 1,
        },
      },
    },
  }

  return <Bar data={chartData} options={options} />
}

export function PnLDistributionChart({ data }) {
  if (!data || data.length === 0) {
    return <p className="text-gray-500">No data available</p>
  }

  // Bucket PnL into ranges
  const bucketSize = 500
  const buckets = {}

  data.forEach(item => {
    const bucket = Math.floor(item.pnl / bucketSize) * bucketSize
    buckets[bucket] = (buckets[bucket] || 0) + 1
  })

  const sortedBuckets = Object.keys(buckets).sort((a, b) => a - b)

  const chartData = {
    labels: sortedBuckets.map(b => `${parseInt(b)}-${parseInt(b) + bucketSize}`),
    datasets: [
      {
        label: 'Trade Count',
        data: sortedBuckets.map(b => buckets[b]),
        backgroundColor: sortedBuckets.map(b => parseInt(b) >= 0 ? '#10b981' : '#ef4444'),
        borderColor: '#374151',
        borderWidth: 1,
      },
    ],
  }

  const options = {
    responsive: true,
    maintainAspectRatio: true,
    plugins: {
      legend: {
        position: 'top',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        ticks: {
          stepSize: 1,
        },
      },
    },
  }

  return <Bar data={chartData} options={options} />
}

export function StrategyPerformanceChart({ data }) {
  if (!data || data.length === 0) {
    return <p className="text-gray-500">No data available</p>
  }

  const chartData = {
    labels: data.map(d => d.strategy),
    datasets: [
      {
        label: 'Win Rate %',
        data: data.map(d => d.win_rate),
        backgroundColor: '#10b981',
        borderColor: '#059669',
        borderWidth: 1,
      },
    ],
  }

  const options = {
    responsive: true,
    maintainAspectRatio: true,
    plugins: {
      legend: {
        position: 'top',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
      },
    },
  }

  return <Bar data={chartData} options={options} />
}
