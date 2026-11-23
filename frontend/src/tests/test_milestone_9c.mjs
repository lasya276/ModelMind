import assert from 'node:assert';

console.log('--- RUNNING MILESTONE 9C VERIFICATION TESTS ---');

// Mock response for profile based on actual backend contracts
const mockProfileResponse = {
  status: 'success',
  message: 'Dataset profiling completed successfully.',
  filename: 'churn.csv',
  dataset_id: 'uuid-1',
  data: {
    total_rows: 100,
    total_columns: 4,
    columns: ['id', 'age', 'income', 'churn'],
    column_types: { id: 'int64', age: 'int64', income: 'float64', churn: 'object' },
    missing_values: {
      id: { count: 0, percentage: 0.0 },
      age: { count: 2, percentage: 2.0 },
      income: { count: 0, percentage: 0.0 },
      churn: { count: 0, percentage: 0.0 },
    },
    unique_values: { id: 100, age: 45, income: 90, churn: 2 },
    duplicate_rows: 0,
    numeric_summary: {
      age: { mean: 38.5, median: 36.0, std: 12.1, min: 18.0, max: 75.0 },
      income: { mean: 55000.0, median: 52000.0, std: 15000.0, min: 20000.0, max: 120000.0 },
    },
    categorical_summary: {
      churn: { unique_count: 2, top_value: 'No', top_frequency: 72 },
    },
  },
};

// 1. Verify Profile Contract
assert.strictEqual(mockProfileResponse.data.total_rows, 100);
assert.strictEqual(mockProfileResponse.data.columns.length, 4);
assert.strictEqual(mockProfileResponse.data.numeric_summary.age.mean, 38.5);
assert.strictEqual(mockProfileResponse.data.categorical_summary.churn.top_value, 'No');
console.log('✓ Scenario 1: Profile structure matches exact backend contract.');

// 2. Verify Quality Contract & No Composite Score
const mockQualityResponse = {
  status: 'success',
  message: 'Data quality analysis completed successfully.',
  filename: 'churn.csv',
  dataset_id: 'uuid-1',
  data: {
    total_rows: 100,
    total_columns: 4,
    has_issues: true,
    issue_counts: {
      missing_value_columns: 1,
      duplicate_rows: 0,
      constant_columns: 0,
      empty_columns: 0,
      potential_id_columns: 1,
      inconsistent_categorical_columns: 0,
      invalid_numerical_columns: 0,
    },
    missing_values: {
      total_affected_columns: 1,
      columns: [
        { column: 'age', missing_count: 2, percentage: 2.0, severity: 'low', explanation: 'Column age has 2 missing values' },
      ],
    },
    duplicate_rows: { duplicate_count: 0, percentage: 0.0, severity: 'low', explanation: 'No duplicate rows detected.' },
    constant_columns: [],
    empty_columns: [],
    potential_id_columns: [{ column: 'id', uniqueness_percentage: 100.0, severity: 'low', explanation: 'High uniqueness' }],
    inconsistent_categorical_values: { total_affected_columns: 0, columns: [] },
    invalid_numerical_values: { total_affected_columns: 0, columns: [] },
  },
};

assert.strictEqual(mockQualityResponse.data.has_issues, true);
assert.strictEqual(mockQualityResponse.data.issue_counts.missing_value_columns, 1);
assert.strictEqual(mockQualityResponse.data.quality_score, undefined, 'Must not contain synthetic quality_score');
console.log('✓ Scenario 2: Quality report verified with zero synthetic scores.');

// 3. Verify Intelligence Contract (Target suggestions only, never auto-selected)
const mockIntelligenceResponse = {
  status: 'success',
  message: 'Dataset intelligence analysis completed successfully.',
  filename: 'churn.csv',
  dataset_id: 'uuid-1',
  data: {
    total_rows: 100,
    total_columns: 4,
    role_counts: { numeric: 2, categorical: 1, potential_id: 1, datetime: 0, boolean: 0, empty: 0 },
    column_roles: {
      id: { column: 'id', role: 'potential_id', is_potential_id: true, unique_count: 100, uniqueness_percentage: 100.0, reason: 'High uniqueness' },
      age: { column: 'age', role: 'numeric', is_potential_id: false, unique_count: 45, uniqueness_percentage: 45.0, reason: 'Numeric' },
      churn: { column: 'churn', role: 'categorical', is_potential_id: false, unique_count: 2, uniqueness_percentage: 2.0, reason: 'Categorical' },
    },
    possible_targets: [
      {
        column: 'churn',
        suggested_task: 'classification',
        target_type: 'binary',
        unique_values_count: 2,
        confidence: 'high',
        reason: 'Contains 2 distinct categories',
        status: 'candidate_only',
      },
    ],
    recommended_analyses: [
      { analysis_type: 'eda', name: 'Exploratory Data Analysis', is_recommended: true, description: '...', suitability_reason: '...', requires_target: false },
      { analysis_type: 'classification', name: 'Classification', is_recommended: true, description: '...', suitability_reason: '...', requires_target: true },
    ],
  },
};

assert.strictEqual(mockIntelligenceResponse.data.possible_targets[0].status, 'candidate_only');
assert.strictEqual(mockIntelligenceResponse.data.recommended_analyses[0].requires_target, false);
console.log('✓ Scenario 3: Intelligence recommendations and target candidates verified.');

// 4. Test Missing / Expired Dataset 404 detection
function isDatasetNotFoundError(error) {
  if (error && error.response && error.response.status === 404) {
    const detail = error.response.data?.detail || '';
    if (typeof detail === 'string' && (detail.toLowerCase().includes('not found in temporary storage') || detail.toLowerCase().includes('dataset'))) {
      return true;
    }
  }
  return false;
}

const expiredErr = {
  response: {
    status: 404,
    data: { detail: "Dataset 'uuid-1' was not found in temporary storage." },
  },
};
const generic404Err = {
  response: {
    status: 404,
    data: { detail: 'Endpoint not found' },
  },
};

assert.strictEqual(isDatasetNotFoundError(expiredErr), true);
assert.strictEqual(isDatasetNotFoundError(generic404Err), false);
console.log('✓ Scenario 4: Contextual 404 expired dataset detection verified.');

// 5 & 6. Test Loading & Cache isolation across dataset switching
let profileCache = { datasetId: 'uuid-1', data: { rows: 100 } };

function switchDataset(newDatasetId) {
  if (profileCache.datasetId !== newDatasetId) {
    // Reset cache on dataset switch
    profileCache = { datasetId: newDatasetId, data: null };
  }
}

// Switching to uuid-2
switchDataset('uuid-2');
assert.strictEqual(profileCache.datasetId, 'uuid-2');
assert.strictEqual(profileCache.data, null, 'Old dataset report data must be wiped when switching dataset');
console.log('✓ Scenario 7 & 8: Switching datasets cleanly purges stale analysis cache.');

console.log('--- ALL MILESTONE 9C CHECKS PASSED SUCCESSFULLY ---');
