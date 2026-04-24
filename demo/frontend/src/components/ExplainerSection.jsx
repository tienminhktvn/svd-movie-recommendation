/* ─── ExplainerSection.jsx
   Giải thích ngắn gọn quy trình SVD theo 3 bước ─── */

const steps = [
  {
    num: '01',
    icon: '📊',
    title: 'Ma trận Rating',
    desc: 'Dataset gồm 610 users × 9.742 phim. Áp dụng Mean-Centering để chuẩn hóa dữ liệu.',
  },
  {
    num: '02',
    icon: '🔬',
    title: 'Phân rã SVD',
    desc: 'Phân rã A = U·S·Vᵀ. Ma trận U chứa đặc trưng tiềm ẩn của từng bộ phim theo k chiều.',
  },
  {
    num: '03',
    icon: '🎯',
    title: 'Dự đoán bằng Pseudo-inverse',
    desc: 'Giải Ax = b (pinv) để tìm vector sở thích. Dot-product với đặc trưng phim → rating dự đoán.',
  },
]

export default function ExplainerSection() {
  return (
    <section className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
      {steps.map((s) => (
        <div
          key={s.num}
          className="glass-card p-5 flex flex-col gap-2 group
                     hover:border-indigo-500/40 transition-colors duration-300"
        >
          <div className="flex items-center gap-3">
            <span className="text-2xl">{s.icon}</span>
            <span className="text-xs font-bold text-indigo-400 tracking-widest">{s.num}</span>
          </div>
          <h3 className="font-['Space_Grotesk'] font-semibold text-white text-base">{s.title}</h3>
          <p className="text-slate-400 text-sm leading-relaxed">{s.desc}</p>
        </div>
      ))}
    </section>
  )
}
