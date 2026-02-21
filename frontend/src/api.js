const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function predictImage(file) {
  const fd = new FormData();
  fd.append("file", file);

  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    body: fd,
  });

  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
