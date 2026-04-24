/* ─── ResultChart.jsx — full-width per movie, no error, larger chart ─── */
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, LabelList,
} from 'recharts'

const USER_LABELS = ['A', 'B', 'C', 'D', 'E']

const TEST_INFO = {
  6377: { name: 'Finding Nemo',  poster: '/Finding Nemo.jpg' },
  5618: { name: 'Spirited Away', poster: '/Spirited Away.jpg' },
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-white border border-gray-200 rounded-xl shadow-lg p-4 text-sm min-w-[150px]">
      <p className="font-bold text-gray-700 mb-2">User {label}</p>
      {payload.map(p => (
        <div key={p.dataKey} className="flex items-center gap-2 mb-1">
          <span className="w-3 h-3 rounded-sm shrink-0" style={{ background: p.fill }} />
          <span className="text-gray-500 text-xs">{p.dataKey}:</span>
          <span className="font-bold text-gray-900">{Number(p.value).toFixed(1)} ★</span>
        </div>
      ))}
    </div>
  )
}

function CustomLegend() {
  return (
    <div className="flex items-center justify-center gap-8 mt-3">
      <div className="flex items-center gap-2">
        <span className="w-5 h-5 rounded" style={{ background: '#3b82f6' }} />
        <span className="text-sm font-semibold text-gray-600">Dự đoán</span>
      </div>
      <div className="flex items-center gap-2">
        <span className="w-5 h-5 rounded" style={{ background: '#10b981' }} />
        <span className="text-sm font-semibold text-gray-600">Thực tế</span>
      </div>
    </div>
  )
}

function MovieChart({ name, poster, chartData }) {
  return (
    <div className="border border-gray-200 rounded-xl p-6 bg-white shadow-sm">
      {/* Header */}
      <div className="flex items-center gap-4 mb-6">
        {poster && (
          <img
            src={poster} alt={name}
            className="w-12 h-[68px] object-cover rounded-lg shadow-md border border-gray-200"
          />
        )}
        <div>
          <h3 className="font-bold text-gray-900 text-lg leading-tight">{name}</h3>
          <p className="text-sm text-gray-400 mt-0.5">Dự đoán vs Thực tế</p>
        </div>
      </div>

      {/* Full-width tall chart */}
      <ResponsiveContainer width="100%" height={320}>
        <BarChart data={chartData} barCategoryGap="35%" barGap={6}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" vertical={false} />

          <XAxis
            dataKey="user"
            tick={{ fill: '#111827', fontSize: 16, fontWeight: 700 }}
            axisLine={{ stroke: '#d1d5db' }}
            tickLine={false}
          />
          <YAxis
            domain={[0, 5.8]}
            ticks={[0, 1, 2, 3, 4, 5]}
            tick={{ fill: '#6b7280', fontSize: 14 }}
            axisLine={false}
            tickLine={false}
            tickFormatter={v => `${v}★`}
            width={40}
          />

          <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(0,0,0,0.04)' }} />
          <Legend content={<CustomLegend />} />

          <Bar dataKey="Dự đoán" fill="#3b82f6" radius={[6, 6, 0, 0]} maxBarSize={56}>
            <LabelList
              dataKey="Dự đoán"
              position="top"
              formatter={v => v?.toFixed(1)}
              style={{ fill: '#1d4ed8', fontSize: 14, fontWeight: 700 }}
            />
          </Bar>

          <Bar dataKey="Thực tế" fill="#10b981" radius={[6, 6, 0, 0]} maxBarSize={56}>
            <LabelList
              dataKey="Thực tế"
              position="top"
              formatter={v => v?.toFixed(1)}
              style={{ fill: '#065f46', fontSize: 14, fontWeight: 700 }}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

export default function ResultChart({ results }) {
  if (!results?.length) return null

  const movieIds   = [6377, 5618]
  const chartsData = movieIds.map(movieId => {
    const info = TEST_INFO[movieId]
    const data = results.map((r, ri) => ({
      user:       USER_LABELS[ri] ?? `U${ri}`,
      'Dự đoán': r.testResults.find(t => t.movieId === movieId)?.predicted ?? 0,
      'Thực tế':  r.testResults.find(t => t.movieId === movieId)?.actual    ?? 0,
    }))
    return { movieId, ...info, data }
  })

  return (
    <div className="mt-2">
      <div className="flex items-center gap-3 mb-5">
        <h2 className="text-sm font-bold text-gray-600 uppercase tracking-widest">
          Kết quả dự đoán
        </h2>
        <div className="flex-1 h-px bg-gray-200" />
      </div>

      {/* Stack vertically — each movie takes full width */}
      <div className="flex flex-col gap-6">
        {chartsData.map(c => (
          <MovieChart key={c.movieId} name={c.name} poster={c.poster} chartData={c.data} />
        ))}
      </div>
    </div>
  )
}
