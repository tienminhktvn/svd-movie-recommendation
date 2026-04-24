export default function Header() {
  return (
    <header className="pt-12 pb-8 text-center">
      {/* Badge */}
      <div className="inline-flex items-center gap-2 px-4 py-1.5 mb-6
                      bg-indigo-500/10 border border-indigo-500/30 rounded-full
                      text-xs font-semibold text-indigo-300 uppercase tracking-widest">
        <span className="relative flex h-2 w-2">
          <span className="ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-60" />
          <span className="relative inline-flex h-2 w-2 rounded-full bg-indigo-500" />
        </span>
        Live Demo · Hệ Thống Gợi Ý Phim
      </div>

      <h1 className="font-['Space_Grotesk'] text-5xl md:text-6xl font-bold mb-4 leading-tight">
        <span className="gradient-text">SVD Movie</span>
        <br />
        <span className="text-white">Recommender</span>
      </h1>

      <p className="max-w-2xl mx-auto text-slate-400 text-base md:text-lg leading-relaxed">
        Minh họa thuật toán{' '}
        <span className="text-indigo-300 font-semibold">Singular Value Decomposition (SVD)</span>{' '}
        dùng ma trận giả nghịch đảo (Pseudo-inverse) để dự đoán rating phim
        từ 5 bộ phim "mồi" và 2 bộ phim ẩn.
      </p>
    </header>
  )
}
