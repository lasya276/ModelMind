import axios from 'axios';

/**
 * Pre-configured Axios instance for ModelMind API calls.
 * Requests to /api are proxied by Vite to the backend (http://127.0.0.1:8000).
 */
const apiClient = axios.create({
  baseURL: '/api',
  timeout: 60000, // 60s timeout for ML training tasks
  headers: {
    'Accept': 'application/json',
  },
});

/**
 * Checks if an API error corresponds specifically to a dataset not found/expired in temporary storage.
 * Backend returns 404 with detail matching: "Dataset '...' was not found in temporary storage."
 */
export function isDatasetNotFoundError(error) {
  if (error && error.response && error.response.status === 404) {
    const detail = error.response.data?.detail || '';
    if (typeof detail === 'string' && (detail.toLowerCase().includes('not found in temporary storage') || detail.toLowerCase().includes('dataset'))) {
      return true;
    }
  }
  return false;
}

/**
 * Extracts a human-readable error message from an API response error.
 */
export function getErrorMessage(error, defaultMessage = 'An unexpected error occurred.') {
  if (error && error.response) {
    const detail = error.response.data?.detail;
    if (typeof detail === 'string') {
      return detail;
    }
    if (Array.isArray(detail)) {
      // FastAPI 422 validation errors
      return detail.map((d) => d.msg || JSON.stringify(d)).join('; ');
    }
    if (error.response.status === 413) {
      return 'File size exceeds the 50 MB limit. Please upload a smaller CSV.';
    }
    if (error.response.status === 404) {
      return 'The requested resource or dataset was not found.';
    }
    if (error.response.status === 500) {
      return 'Server error occurred during processing. Please try again.';
    }
  }
  return error?.message || defaultMessage;
}

export default apiClient;
