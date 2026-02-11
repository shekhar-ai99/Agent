// Button component
export function Button({ children, variant = 'primary', size = 'md', className = '', ...props }) {
  const baseStyles = 'font-medium rounded-lg transition inline-flex items-center justify-center'
  
  const variants = {
    primary: 'bg-trading-600 text-white hover:bg-trading-700',
    secondary: 'bg-gray-200 text-gray-900 hover:bg-gray-300',
    outline: 'border-2 border-trading-600 text-trading-600 hover:bg-trading-50',
    danger: 'bg-red-600 text-white hover:bg-red-700',
    success: 'bg-green-600 text-white hover:bg-green-700',
  }

  const sizes = {
    sm: 'px-3 py-1 text-sm',
    md: 'px-4 py-2 text-base',
    lg: 'px-6 py-3 text-lg',
  }

  return (
    <button
      className={`${baseStyles} ${variants[variant]} ${sizes[size]} ${className}`}
      {...props}
    >
      {children}
    </button>
  )
}

// Card component
export function Card({ children, className = '' }) {
  return (
    <div className={`bg-white rounded-lg shadow-md p-6 ${className}`}>
      {children}
    </div>
  )
}

// Metric Card
export function MetricCard({ label, value, unit = '', icon = null, trend = null }) {
  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-sm text-gray-600 font-medium">{label}</p>
          <p className="mt-2 text-3xl font-bold text-gray-900">
            {value}
            <span className="text-lg text-gray-500 ml-1">{unit}</span>
          </p>
          {trend && (
            <p className={`mt-2 text-sm ${trend > 0 ? 'text-green-600' : 'text-red-600'}`}>
              {trend > 0 ? '↑' : '↓'} {Math.abs(trend)}% from previous
            </p>
          )}
        </div>
        {icon && (
          <div className="text-3xl text-trading-600">
            {icon}
          </div>
        )}
      </div>
    </div>
  )
}

// Loading Spinner
export function Spinner() {
  return (
    <div className="flex items-center justify-center">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-trading-600"></div>
    </div>
  )
}

// Alert
export function Alert({ message, type = 'info', onClose = null }) {
  const bgColors = {
    info: 'bg-blue-50 border-blue-200',
    success: 'bg-green-50 border-green-200',
    warning: 'bg-yellow-50 border-yellow-200',
    error: 'bg-red-50 border-red-200',
  }

  const textColors = {
    info: 'text-blue-800',
    success: 'text-green-800',
    warning: 'text-yellow-800',
    error: 'text-red-800',
  }

  return (
    <div className={`border rounded-lg p-4 ${bgColors[type]}`}>
      <div className="flex items-start justify-between">
        <p className={`${textColors[type]} font-medium`}>{message}</p>
        {onClose && (
          <button onClick={onClose} className="text-gray-500 hover:text-gray-700">
            ✕
          </button>
        )}
      </div>
    </div>
  )
}

// Select Component
export function Select({ label, value, onChange, options, disabled = false, required = false }) {
  return (
    <div className="flex flex-col">
      {label && (
        <label className="text-sm font-medium text-gray-700 mb-2">
          {label} {required && <span className="text-red-500">*</span>}
        </label>
      )}
      <select
        value={value}
        onChange={onChange}
        disabled={disabled}
        className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-trading-600 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
      >
        <option value="">-- Select --</option>
        {options.map(opt => (
          <option key={opt.id} value={opt.id}>
            {opt.name || opt.label}
          </option>
        ))}
      </select>
    </div>
  )
}

// Input Component
export function Input({ label, type = 'text', value, onChange, placeholder = '', disabled = false, required = false }) {
  return (
    <div className="flex flex-col">
      {label && (
        <label className="text-sm font-medium text-gray-700 mb-2">
          {label} {required && <span className="text-red-500">*</span>}
        </label>
      )}
      <input
        type={type}
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        disabled={disabled}
        className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-trading-600 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
      />
    </div>
  )
}

// Badge component
export function Badge({ children, variant = 'primary' }) {
  const variants = {
    primary: 'bg-trading-100 text-trading-800',
    success: 'bg-green-100 text-green-800',
    warning: 'bg-yellow-100 text-yellow-800',
    error: 'bg-red-100 text-red-800',
    gray: 'bg-gray-100 text-gray-800',
  }

  return (
    <span className={`inline-block px-3 py-1 rounded-full text-xs font-medium ${variants[variant]}`}>
      {children}
    </span>
  )
}

// Table component
export function Table({ columns, data, loading = false, onRowClick = null }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-200 bg-gray-50">
            {columns.map(col => (
              <th key={col.id} className="px-6 py-3 text-left font-medium text-gray-700">
                {col.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {loading ? (
            <tr>
              <td colSpan={columns.length} className="px-6 py-4 text-center">
                <Spinner />
              </td>
            </tr>
          ) : data.length === 0 ? (
            <tr>
              <td colSpan={columns.length} className="px-6 py-4 text-center text-gray-500">
                No data available
              </td>
            </tr>
          ) : (
            data.map((row, idx) => (
              <tr 
                key={idx} 
                className="border-b border-gray-200 hover:bg-gray-50 cursor-pointer"
                onClick={() => onRowClick && onRowClick(row)}
              >
                {columns.map(col => (
                  <td key={col.id} className="px-6 py-4">
                    {col.render ? col.render(row) : row[col.id]}
                  </td>
                ))}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  )
}

// Empty State
export function EmptyState({ title, description, action = null }) {
  return (
    <div className="text-center py-12">
      <h3 className="mt-2 text-lg font-medium text-gray-900">{title}</h3>
      <p className="mt-1 text-sm text-gray-500">{description}</p>
      {action && <div className="mt-6">{action}</div>}
    </div>
  )
}
