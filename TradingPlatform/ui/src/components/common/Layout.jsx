import { Link } from 'react-router-dom'

function Layout({ children }) {
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex justify-between items-center">
            <Link to="/" className="flex items-center gap-2">
              <div className="w-8 h-8 bg-trading-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-lg">T</span>
              </div>
              <h1 className="text-2xl font-bold text-gray-900">TradingPlatform</h1>
            </Link>
            
            <div className="flex gap-4">
              <Link 
                to="/" 
                className="px-3 py-2 rounded-lg text-sm font-medium text-gray-700 hover:bg-gray-100 transition"
              >
                Configure
              </Link>
              <Link 
                to="/rankings" 
                className="px-3 py-2 rounded-lg text-sm font-medium text-gray-700 hover:bg-gray-100 transition"
              >
                Rankings
              </Link>
            </div>
          </div>
        </nav>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {children}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <p className="text-center text-sm text-gray-600">
            © 2026 TradingPlatform. Built with React + Tailwind.
          </p>
        </div>
      </footer>
    </div>
  )
}

export default Layout
