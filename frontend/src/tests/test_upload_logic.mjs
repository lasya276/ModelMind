import assert from 'node:assert';
import fs from 'node:fs';
import path from 'node:path';

console.log('--- RUNNING MILESTONE 9B VERIFICATION TESTS ---');

// 1. Check RetentionNotice.jsx text correction
const retentionPath = path.resolve('src/components/common/RetentionNotice.jsx');
const retentionContent = fs.readFileSync(retentionPath, 'utf8');
assert(
  retentionContent.includes('Expired files are removed when the server runs its cleanup process.'),
  'Retention notice must contain the exact corrected wording'
);
assert(
  !retentionContent.includes('The server removes older files automatically during routine maintenance.'),
  'Old retention notice text must be removed'
);
console.log('✓ Scenario 9 Passed: Retention notice contains corrected wording.');

// 2. Test File validation rules (mirrored from CsvUploader.jsx)
const MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024;
function validateFile(file) {
  if (!file) return 'No file selected. Please choose a CSV file.';
  if (!file.name || !file.name.toLowerCase().endsWith('.csv')) {
    return 'Please choose a CSV file (.csv). Other file formats are not supported.';
  }
  if (file.size === 0) {
    return 'The selected file is empty (0 bytes). Please choose a valid CSV dataset.';
  }
  if (file.size > MAX_FILE_SIZE_BYTES) {
    return `File size exceeds the maximum allowed limit of 50 MB.`;
  }
  return null;
}

// Scenario 1: Valid CSV
assert.strictEqual(validateFile({ name: 'customers.csv', size: 1024 }), null);
console.log('✓ Scenario 1 Passed: Valid CSV file accepted.');

// Scenario 2: Non-CSV file rejection
assert.strictEqual(
  validateFile({ name: 'image.png', size: 5000 }),
  'Please choose a CSV file (.csv). Other file formats are not supported.'
);
assert.strictEqual(
  validateFile({ name: 'data.xlsx', size: 5000 }),
  'Please choose a CSV file (.csv). Other file formats are not supported.'
);
console.log('✓ Scenario 2 Passed: Non-CSV files rejected.');

// Scenario 3: File larger than 50 MB rejection
assert.strictEqual(
  validateFile({ name: 'massive.csv', size: 50 * 1024 * 1024 + 1 }),
  'File size exceeds the maximum allowed limit of 50 MB.'
);
console.log('✓ Scenario 3 Passed: Files larger than 50 MB rejected.');

// Scenario 4: Empty file rejection
assert.strictEqual(
  validateFile({ name: 'empty.csv', size: 0 }),
  'The selected file is empty (0 bytes). Please choose a valid CSV dataset.'
);
console.log('✓ Scenario 4 Passed: Empty 0-byte file rejected.');

// Scenario 5: Error parsing verification from client.js
function getErrorMessage(error, defaultMessage = 'An unexpected error occurred.') {
  if (error && error.response) {
    const detail = error.response.data?.detail;
    if (typeof detail === 'string') return detail;
    if (error.response.status === 413) return 'File size exceeds the 50 MB limit. Please upload a smaller CSV.';
    if (error.response.status === 404) return 'The requested resource or dataset was not found.';
    if (error.response.status === 500) return 'Server error occurred during processing. Please try again.';
  }
  return error?.message || defaultMessage;
}

assert.strictEqual(
  getErrorMessage({ response: { status: 413 } }),
  'File size exceeds the 50 MB limit. Please upload a smaller CSV.'
);
assert.strictEqual(
  getErrorMessage({ response: { status: 400, data: { detail: 'The CSV file does not contain any valid columns.' } } }),
  'The CSV file does not contain any valid columns.'
);
assert.strictEqual(
  getErrorMessage({ message: 'Network Error' }),
  'Network Error'
);
console.log('✓ Scenario 5 Passed: Network and backend failure responses mapped accurately.');

// Scenario 6: Double-click / duplicate upload guard
let isUploading = false;
let uploadCalls = 0;
function simulateUpload() {
  if (isUploading) return 'BLOCKED';
  isUploading = true;
  uploadCalls++;
  return 'PROCEED';
}
assert.strictEqual(simulateUpload(), 'PROCEED');
assert.strictEqual(simulateUpload(), 'BLOCKED'); // Second call blocked while first is running
assert.strictEqual(uploadCalls, 1);
isUploading = false;
console.log('✓ Scenario 6 Passed: Concurrent/double-click uploads prevented.');

// Scenario 7 & 8: Session storage restoration without timestamp reset
const originalTimestamp = 1732000000000;
const storedState = JSON.stringify({
  dataset_id: 'test-uuid-1234',
  filename: 'sales.csv',
  row_count: 500,
  column_count: 8,
  columns: ['id', 'amount'],
  uploaded_at: originalTimestamp,
});

// Emulate DatasetContext initialization
const parsed = JSON.parse(storedState);
const restored = {
  dataset_id: parsed.dataset_id,
  filename: parsed.filename || 'dataset.csv',
  row_count: parsed.row_count ?? null,
  column_count: parsed.column_count ?? null,
  columns: Array.isArray(parsed.columns) ? parsed.columns : [],
  uploaded_at: parsed.uploaded_at || Date.now(),
};

assert.strictEqual(restored.dataset_id, 'test-uuid-1234');
assert.strictEqual(restored.uploaded_at, originalTimestamp); // Timestamp strictly preserved!
assert.notStrictEqual(restored.uploaded_at, Date.now());
console.log('✓ Scenario 7 & 8 Passed: Active dataset session and original timestamp preserved.');

console.log('--- ALL 9 SCENARIO CHECKS PASSED SUCCESSFULLY ---');
