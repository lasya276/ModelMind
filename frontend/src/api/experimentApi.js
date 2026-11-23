import apiClient from './client';
import { buildExperimentRequest } from '../utils/experimentUtils';

export async function runExperiment(type, datasetId, config, signal) {
  const { endpoint, payload } = buildExperimentRequest(type, datasetId, config);
  const response = await apiClient.post(endpoint, payload, { signal });
  return response.data;
}
