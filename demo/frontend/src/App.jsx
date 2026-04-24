import { useState, useRef } from 'react'
import RatingTable from './components/RatingTable'
import PredictButton from './components/PredictButton'
import ResultChart from './components/ResultChart'

export default function App() {
  const [initialData, setInitialData] = useState(null)
  const [results, setResults]         = useState(null)
  const [loading, setLoading]         = useState(false)
  const [error, setError]             = useState(null)
  const fetchedRef = useRef(false)

  if (!fetchedRef.current) {
    fetchedRef.current = true
    fetch('/api/users-initial')
      .then(r => { if (!r.ok) throw new Error(r.statusText); return r.json() })
      .then(d => setInitialData(d))
      .catch(e => setError(e.message))
  }

  const handlePredict = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch('/api/predict', { method: 'POST' })
      if (!res.ok) throw new Error(res.statusText)
      const data = await res.json()
      setResults(data.results)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-6xl mx-auto px-6 py-10">

      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-600 text-sm">
          Lỗi: {error}
        </div>
      )}

      {!initialData && !error && (
        <div className="flex items-center justify-center gap-2 text-gray-400 text-sm py-24">
          <span className="spinner" /> Đang tải dữ liệu…
        </div>
      )}

      {initialData && (
        <>
          {/* Rating table – full overflow scroll on small screens */}
          <div className="mb-7 overflow-x-auto">
            <RatingTable data={initialData} results={results} />
          </div>

          {/* Predict button */}
          <div className="flex justify-center mb-10">
            <PredictButton loading={loading} done={!!results} onClick={handlePredict} />
          </div>
        </>
      )}

      {/* Charts */}
      {results && (
        <div className="anim-fade-up">
          <ResultChart results={results} />
        </div>
      )}
    </div>
  )
}
