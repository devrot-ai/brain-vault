// BrainVuln frontend configuration (static site on Vercel).
// The analyze section reads window.BRAINVULN_API_BASE from this file.
// Set it to the deployed inference-service URL, e.g.
//   window.BRAINVULN_API_BASE = "https://brainvuln-api.onrender.com";
// or leave it empty ("") — the site then shows an honest
// "inference backend not configured" message instead of faking results.
window.BRAINVULN_API_BASE = "";
window.BRAINVULN_MAX_UPLOAD_MB = 200;
