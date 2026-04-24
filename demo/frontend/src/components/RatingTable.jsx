const POSTER = {
  'Toy Story':                '/Toy Story.jpg',
  'Spider-Man':               '/Spider-Man.jpg',
  'Despicable Me':            '/Despicable Me.webp',
  'Harry Potter':             '/Harry Potter.jpg',
  'Pirates of the Caribbean': '/Pirates of the Caribbean.webp',
  'Finding Nemo':             '/Finding Nemo.jpg',
  'Spirited Away':            '/Spirited Away.jpg',
}

const USER_LABELS  = ['A', 'B', 'C', 'D', 'E']
const AVATAR_STYLE = [
  'bg-indigo-100 text-indigo-700 ring-indigo-300',
  'bg-rose-100   text-rose-700   ring-rose-300',
  'bg-amber-100  text-amber-700  ring-amber-300',
  'bg-emerald-100 text-emerald-700 ring-emerald-300',
  'bg-sky-100    text-sky-700    ring-sky-300',
]

/** Anchor rating */
function RatingCell({ rating }) {
  if (rating == null) return <span className="text-gray-300 text-xs">—</span>
  const display = rating % 1 === 0 ? rating.toFixed(0) : rating.toFixed(1)
  return (
    <span className="inline-flex items-baseline gap-0.5 font-bold text-gray-800 text-sm">
      {display}
      <span className="text-yellow-400 text-xs">★</span>
    </span>
  )
}

/** Placeholder before prediction */
function PlaceholderCell() {
  return <span className="text-gray-300 text-sm font-bold">?</span>
}

/** Result: only show predicted value */
function ResultCell({ predicted }) {
  return (
    <span className="inline-flex items-baseline gap-0.5 px-2 py-0.5 rounded
                     bg-blue-50 border border-blue-200 text-blue-800 font-bold text-sm">
      {predicted?.toFixed(1)}
      <span className="text-yellow-400 text-xs">★</span>
    </span>
  )
}

export default function RatingTable({ data, results }) {
  const { users, movies } = data

  const resultMap = {}
  if (results) {
    results.forEach(r => {
      resultMap[r.userId] = {}
      r.testResults.forEach(t => { resultMap[r.userId][t.movieId] = t })
    })
  }

  const anchors   = movies.filter(m => m.role === 'anchor')
  const tests     = movies.filter(m => m.role === 'test')
  const allMovies = [...anchors, ...tests]

  return (
    <div className="overflow-x-auto rounded-xl border border-gray-300 shadow-sm">
      <table className="w-full border-collapse text-sm" style={{ borderCollapse: 'collapse' }}>

        {/* Movie header row */}
        <thead>
          <tr className="border-b-2 border-gray-300">
            <th className="py-2 px-4 text-left text-gray-500 font-semibold text-xs
                           bg-gray-50 border-r-2 border-gray-300 w-20">
              User
            </th>

            {allMovies.map((m, i) => {
              const isLastAnch = i === anchors.length - 1
              const isTest     = m.role === 'test'
              return (
                <th
                  key={m.movieId}
                  className={`py-2 px-2 text-center
                    ${isLastAnch ? 'border-r-2 border-gray-400' : 'border-r border-gray-200'}
                    ${isTest ? 'bg-blue-50' : 'bg-white'}
                  `}
                >
                  {POSTER[m.name] ? (
                    <img
                      src={POSTER[m.name]}
                      alt={m.name}
                      className="w-10 h-14 object-cover rounded-md mx-auto mb-1
                                 shadow-md border border-gray-200"
                    />
                  ) : (
                    <div className="w-10 h-14 bg-gray-100 rounded-md mx-auto mb-1" />
                  )}
                  <div className="text-[11px] font-bold text-gray-700 leading-tight
                                  max-w-[64px] mx-auto text-center">
                    {m.name}
                  </div>
                </th>
              )
            })}
          </tr>
        </thead>

        {/* Body */}
        <tbody>
          {users.map((user, ri) => {
            const label     = USER_LABELS[ri] ?? `U${ri}`
            const avatarCls = AVATAR_STYLE[ri] ?? 'bg-gray-100 text-gray-700 ring-gray-300'

            return (
              <tr
                key={user.userId}
                className={`border-b border-gray-200 hover:bg-yellow-50/20 transition-colors
                  ${ri % 2 === 0 ? 'bg-white' : 'bg-gray-50/50'}`}
              >
                {/* Avatar */}
                <td className="py-3 px-4 border-r-2 border-gray-300 bg-gray-50">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center
                                   font-bold text-sm select-none ring-2 ${avatarCls}`}>
                    {label}
                  </div>
                </td>

                {allMovies.map((m, ci) => {
                  const movieData  = user.movies.find(x => x.movieId === m.movieId)
                  const isTest     = m.role === 'test'
                  const isLastAnch = ci === anchors.length - 1
                  const res        = resultMap[user.userId]?.[m.movieId]

                  return (
                    <td
                      key={m.movieId}
                      className={`py-3 px-2 text-center align-middle
                        ${isLastAnch ? 'border-r-2 border-gray-400' : 'border-r border-gray-200'}
                        ${isTest ? 'bg-blue-50/50' : ''}
                      `}
                    >
                      {!isTest ? (
                        <RatingCell rating={movieData?.rating} />
                      ) : res ? (
                        <ResultCell predicted={res.predicted} />
                      ) : (
                        <PlaceholderCell />
                      )}
                    </td>
                  )
                })}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
