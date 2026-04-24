/* ─── Footer.jsx ─── */
export default function Footer() {
  return (
    <footer className="border-t border-slate-800/60 py-6 mt-4">
      <p className="text-center text-slate-600 text-xs">
        SVD Movie Recommender Demo &nbsp;·&nbsp;
        Dữ liệu: MovieLens 100K &nbsp;·&nbsp;
        Thuật toán: SVD + Pseudo-inverse &nbsp;·&nbsp;
        <span className="text-indigo-400/60">ReactJS · Flask · Recharts</span>
      </p>
    </footer>
  )
}
