const API_BASE = "http://localhost:8000";

export async function predictImage(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    body: formData,
  });

  return res.json();
}