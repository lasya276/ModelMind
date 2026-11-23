import apiClient from './client';

/**
 * Uploads a CSV file to the backend POST /upload endpoint.
 *
 * @param {File} file - The CSV file to upload.
 * @param {Function} onUploadProgress - Callback for tracking upload progress.
 * @param {AbortSignal} signal - Optional abort signal for cancellation.
 * @returns {Promise<Object>} Backend response with dataset metadata and dataset_id.
 */
export async function uploadDataset(file, onUploadProgress, signal) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await apiClient.post('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress: (progressEvent) => {
      if (onUploadProgress && progressEvent.total) {
        const percentCompleted = Math.round(
          (progressEvent.loaded * 100) / progressEvent.total
        );
        onUploadProgress(percentCompleted);
      }
    },
    signal,
  });

  return response.data;
}

/**
 * Generates or retrieves a comprehensive dataset profile via POST /profile.
 *
 * @param {string} datasetId - The UUID of the stored dataset.
 * @param {AbortSignal} signal - Optional abort signal for cancellation.
 * @returns {Promise<Object>} Backend response containing profile report data.
 */
export async function getDatasetProfile(datasetId, signal) {
  const response = await apiClient.post(
    `/profile?dataset_id=${encodeURIComponent(datasetId)}`,
    { dataset_id: datasetId },
    { signal }
  );
  return response.data;
}

/**
 * Performs data quality inspection via POST /quality.
 *
 * @param {string} datasetId - The UUID of the stored dataset.
 * @param {AbortSignal} signal - Optional abort signal for cancellation.
 * @returns {Promise<Object>} Backend response containing quality report data.
 */
export async function getDatasetQuality(datasetId, signal) {
  const response = await apiClient.post(
    `/quality?dataset_id=${encodeURIComponent(datasetId)}`,
    { dataset_id: datasetId },
    { signal }
  );
  return response.data;
}

/**
 * Analyzes dataset intelligence and machine learning suitability via POST /intelligence.
 *
 * @param {string} datasetId - The UUID of the stored dataset.
 * @param {AbortSignal} signal - Optional abort signal for cancellation.
 * @returns {Promise<Object>} Backend response containing intelligence report data.
 */
export async function getDatasetIntelligence(datasetId, signal) {
  const response = await apiClient.post(
    `/intelligence?dataset_id=${encodeURIComponent(datasetId)}`,
    { dataset_id: datasetId },
    { signal }
  );
  return response.data;
}

/**
 * Executes user-approved dataset cleaning operations via POST /clean.
 *
 * @param {string} datasetId - The UUID of the original dataset.
 * @param {Object} operations - Dictionary of cleaning operations.
 * @param {AbortSignal} signal - Optional abort signal for cancellation.
 * @returns {Promise<Object>} Backend response containing cleaned_dataset_id, download_url, and cleaning report.
 */
export async function cleanDataset(datasetId, operations = {}, signal) {
  const response = await apiClient.post(
    `/clean?dataset_id=${encodeURIComponent(datasetId)}`,
    {
      dataset_id: datasetId,
      operations: operations || {},
    },
    { signal }
  );
  return response.data;
}

/**
 * Fetches a bounded preview and total dimensions for a stored cleaned dataset.
 *
 * @param {string} datasetId - The UUID of the cleaned dataset.
 * @param {AbortSignal} signal - Optional abort signal for cancellation.
 * @returns {Promise<Object>} Preview rows, column names, and total dimensions.
 */
export async function getCleanedDatasetPreview(datasetId, signal) {
  const response = await apiClient.get(
    `/clean/preview/${encodeURIComponent(datasetId)}`,
    { signal }
  );
  return response.data;
}

/**
 * Downloads a cleaned dataset CSV file via GET /clean/download/{dataset_id}.
 *
 * @param {string} datasetId - The UUID of the cleaned dataset.
 * @param {string} filename - Desired filename for download.
 * @returns {Promise<void>}
 */
export async function downloadCleanedDataset(datasetId, filename = 'cleaned_dataset.csv') {
  const response = await apiClient.get(`/clean/download/${encodeURIComponent(datasetId)}`, {
    responseType: 'blob',
  });

  const blob = new Blob([response.data], { type: 'text/csv' });
  const downloadUrl = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = downloadUrl;
  link.setAttribute('download', filename);
  document.body.appendChild(link);
  link.click();
  link.parentNode.removeChild(link);
  setTimeout(() => window.URL.revokeObjectURL(downloadUrl), 1000);
}

