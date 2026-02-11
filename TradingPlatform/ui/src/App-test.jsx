function App() {
  return (
    <div style={{ padding: '20px', backgroundColor: '#f0f0f0', minHeight: '100vh' }}>
      <h1 style={{ color: 'blue' }}>React is Working!</h1>
      <p>If you see this blue heading, React is rendering correctly.</p>
      <p>Current time: {new Date().toLocaleTimeString()}</p>
    </div>
  )
}

export default App
