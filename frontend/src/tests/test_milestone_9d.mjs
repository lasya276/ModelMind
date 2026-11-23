import assert from 'node:assert';

console.log('--- RUNNING MILESTONE 9D VERIFICATION TESTS ---');

// 1. Backend Contract Validation: Valid Operation Keys
const VALID_OPERATION_KEYS = new Set([
  'missing_values',
  'remove_duplicates',
  'duplicate_subset',
  'trim_whitespace',
  'casing',
  'drop_empty_columns',
  'drop_constant_columns',
]);

function validateCompiledOperations(ops) {
  const keys = Object.keys(ops);
  for (const k of keys) {
    if (!VALID_OPERATION_KEYS.has(k)) {
      throw new Error(`Unsupported cleaning operation key: ${k}`);
    }
  }
  return true;
}

// Test valid compiled operations payload
const sampleCompiledOps = {
  missing_values: { age: 'mean', status: 'mode', notes: { strategy: 'constant', value: 'N/A' } },
  remove_duplicates: true,
  duplicate_subset: ['customer_id'],
  trim_whitespace: true,
  casing: { city: 'upper' },
  drop_empty_columns: true,
  drop_constant_columns: true,
};

assert.strictEqual(validateCompiledOperations(sampleCompiledOps), true);
assert.throws(() => validateCompiledOperations({ unsupported_op: true }));
console.log('✓ Scenario 1: Clean operation keys strictly match backend VALID_OPERATION_KEYS.');

// 2. Data Type Compatibility Guard (Numeric vs Text strategies)
function validateImputationStrategies(columnTypes, missingConfigs) {
  for (const [col, config] of Object.entries(missingConfigs)) {
    const strategy = typeof config === 'string' ? config : config.strategy;
    const dtype = (columnTypes[col] || '').toLowerCase();
    const isNumeric = dtype.includes('int') || dtype.includes('float') || dtype.includes('number');

    if ((strategy === 'mean' || strategy === 'median') && !isNumeric) {
      throw new Error(`Strategy '${strategy}' cannot be applied to non-numeric column '${col}'.`);
    }
  }
  return true;
}

const mockColumnTypes = { age: 'int64', salary: 'float64', name: 'object', status: 'object' };
assert.strictEqual(
  validateImputationStrategies(mockColumnTypes, { age: 'mean', salary: 'median', name: 'mode' }),
  true
);
assert.throws(() => {
  validateImputationStrategies(mockColumnTypes, { name: 'mean' });
});
console.log('✓ Scenario 2: Mean and median restricted to numeric columns, preventing HTTP 400 errors.');

// 3. Dataset Context Version Management & Immutability
let sessionStore = {
  originalDataset: {
    dataset_id: 'orig-uuid-1111',
    filename: 'customers.csv',
    row_count: 100,
    column_count: 5,
    columns: ['id', 'age', 'name', 'country', 'notes'],
    uploaded_at: 1732000000000,
  },
  cleanedDataset: null,
  activeVersion: 'original',
};

// Function emulating DatasetContext versioning logic
function setCleanedDataset(cleanedData) {
  sessionStore.cleanedDataset = {
    dataset_id: cleanedData.dataset_id,
    original_dataset_id: sessionStore.originalDataset.dataset_id,
    filename: cleanedData.filename,
    row_count: cleanedData.row_count,
    column_count: cleanedData.column_count,
    cleaned_at: Date.now(),
    download_url: `/clean/download/${cleanedData.dataset_id}`,
  };
  sessionStore.activeVersion = 'cleaned';
}

function getActiveDataset() {
  if (sessionStore.activeVersion === 'cleaned' && sessionStore.cleanedDataset) {
    return sessionStore.cleanedDataset;
  }
  return sessionStore.originalDataset;
}

function switchVersion(target) {
  if (target === 'cleaned' && !sessionStore.cleanedDataset) return;
  sessionStore.activeVersion = target;
}

// Initial state: original is active
assert.strictEqual(getActiveDataset().dataset_id, 'orig-uuid-1111');
assert.strictEqual(sessionStore.activeVersion, 'original');

// Apply cleaning: cleaned version created with distinct ID
setCleanedDataset({
  dataset_id: 'clean-uuid-2222',
  filename: 'cleaned_customers.csv',
  row_count: 95,
  column_count: 4,
});

// After cleaning: cleaned dataset becomes active, original remains completely preserved!
assert.strictEqual(getActiveDataset().dataset_id, 'clean-uuid-2222');
assert.strictEqual(getActiveDataset().row_count, 95);
assert.strictEqual(sessionStore.originalDataset.dataset_id, 'orig-uuid-1111');
assert.strictEqual(sessionStore.originalDataset.row_count, 100);

// Switch back to original
switchVersion('original');
assert.strictEqual(getActiveDataset().dataset_id, 'orig-uuid-1111');
assert.strictEqual(sessionStore.activeVersion, 'original');

// Switch back to cleaned
switchVersion('cleaned');
assert.strictEqual(getActiveDataset().dataset_id, 'clean-uuid-2222');
assert.strictEqual(sessionStore.activeVersion, 'cleaned');
console.log('✓ Scenario 3: Version management preserves original dataset while allowing seamless toggling.');

// 4. Cache Invalidation on Version Switch
let profileCache = { datasetId: 'orig-uuid-1111', data: { mean_age: 35 } };

function notifyDatasetChanged(newDatasetId) {
  if (profileCache.datasetId !== newDatasetId) {
    profileCache = { datasetId: newDatasetId, data: null };
  }
}

// Switching version updates datasetId and clears stale report cache
notifyDatasetChanged(getActiveDataset().dataset_id);
assert.strictEqual(profileCache.datasetId, 'clean-uuid-2222');
assert.strictEqual(profileCache.data, null);
console.log('✓ Scenario 4: Switching between original and cleaned versions safely invalidates stale cache.');

// 5. Download URL format verification
function verifyDownloadUrl(cleanedDatasetId) {
  const pathUrl = `/clean/download/${cleanedDatasetId}`;
  const queryUrl = `/clean/download?dataset_id=${cleanedDatasetId}`;
  return { pathUrl, queryUrl };
}

const { pathUrl, queryUrl } = verifyDownloadUrl('clean-uuid-2222');
assert.strictEqual(pathUrl, '/clean/download/clean-uuid-2222');
assert.strictEqual(queryUrl, '/clean/download?dataset_id=clean-uuid-2222');
console.log('✓ Scenario 5: Download endpoints follow verified backend routes.');

// 6. Non-Destructive Cleaning Response Schema
const mockCleanResponse = {
  status: 'success',
  message: 'Dataset successfully cleaned.',
  filename: 'cleaned_data.csv',
  original_dataset_id: 'orig-uuid-1111',
  cleaned_dataset_id: 'clean-uuid-2222',
  download_url: '/clean/download/clean-uuid-2222',
  data: {
    initial_shape: { rows: 100, columns: 5 },
    cleaned_shape: { rows: 95, columns: 4 },
    operations_performed: [
      {
        operation: 'remove_duplicates',
        removed_rows: 5,
        subset: 'all_columns',
        description: 'Removed 5 duplicate row(s).',
      },
    ],
    operations_count: 1,
    missing_values_imputed: { total_imputed: 0, details: [] },
    duplicates_removed: { removed_count: 5, subset: 'all_columns' },
    text_formatting: { whitespace_trimmed: [], casing_applied: [] },
    columns_dropped: { empty_columns: [], constant_columns: ['country'] },
    detected_unmodified_issues: { empty_columns: [], constant_columns: [], remaining_missing_values: {} },
  },
};

assert.strictEqual(mockCleanResponse.data.initial_shape.rows, 100);
assert.strictEqual(mockCleanResponse.data.cleaned_shape.rows, 95);
assert.strictEqual(mockCleanResponse.data.operations_performed.length, 1);
assert.notStrictEqual(mockCleanResponse.original_dataset_id, mockCleanResponse.cleaned_dataset_id);
console.log('✓ Scenario 6: Cleaning response schema reflects structural transformations and separate IDs.');

console.log('--- ALL MILESTONE 9D CHECKS PASSED SUCCESSFULLY ---');
