/* ─── PredictButton.jsx ─── */
export default function PredictButton({ loading, done, onClick }) {
  return (
    <button
      id="btn-predict"
      onClick={onClick}
      disabled={loading}
      className="inline-flex items-center gap-2 px-8 py-3 rounded-lg font-semibold text-sm
                 bg-indigo-600 text-white hover:bg-indigo-700 active:bg-indigo-800
                 disabled:opacity-50 disabled:cursor-not-allowed
                 transition-colors shadow-sm"
    >
      {loading ? (
        <>
          <span className="spinner" /> Đang tính toán SVD…
        </>
      ) : done ? (
        <>🔄 Chạy lại</>
      ) : (
        <>Dự đoán</>
      )}
    </button>
  );
}
