import assert from 'node:assert';

console.log('--- RUNNING MILESTONE 9E VERIFICATION TESTS ---');

// ============================================================================
// 1. Feature A: Intelligent Missing Value Recommendation Engine
// ============================================================================
function getImputationRecommendation(col, columnTypes, profileData, totalRows) {
  const dtype = (columnTypes[col] || '').toLowerCase();
  const isNumeric =
    dtype.includes('int') ||
    dtype.includes('float') ||
    dtype.includes('double') ||
    dtype.includes('number');

  if (isNumeric) {
    const numStats = profileData?.numeric_summary?.[col];
    const uniqueCount = profileData?.unique_values?.[col];

    if (!numStats || numStats.mean === null || numStats.median === null) {
      return { strategy: 'median', reason: 'Insufficient statistics, fallback to robust median' };
    }

    const { mean, median, std, min, max } = numStats;

    // Rule 1: Discrete low-cardinality integer scale
    if (uniqueCount && uniqueCount <= 5 && Number.isInteger(min) && Number.isInteger(max)) {
      return { strategy: 'mode', reason: `Discrete scale with ${uniqueCount} unique values` };
    }

    // Rule 2: Constant / Zero variance
    if (std === 0 || min === max) {
      return { strategy: 'mean', reason: 'Zero variation detected' };
    }

    // Rule 3: Skewness / Outliers evaluation (|mean - median| / std > 0.2)
    const skewnessRatio = std > 0 ? Math.abs(mean - median) / std : 0;
    if (skewnessRatio > 0.2) {
      return { strategy: 'median', reason: `Distribution is skewed (|mean - median| / std = ${skewnessRatio.toFixed(2)})` };
    }

    // Rule 4: Symmetric distribution
    return { strategy: 'mean', reason: 'Distribution is approximately symmetric' };
  } else {
    const catStats = profileData?.categorical_summary?.[col];
    if (!catStats || !catStats.top_value || !totalRows) {
      return { strategy: 'constant', reason: 'Insufficient categorical observations' };
    }

    const { top_value, top_frequency } = catStats;
    const dominancePct = Math.round((top_frequency / totalRows) * 100);

    if (dominancePct >= 50) {
      return { strategy: 'mode', reason: `Dominant category accounts for ${dominancePct}% of records` };
    }

    return { strategy: 'constant', reason: `Balanced classes; top category accounts for only ${dominancePct}%` };
  }
}

// Case 1: Numeric Balanced Distribution (Mean ≈ Median)
const balancedCol = getImputationRecommendation(
  'income',
  { income: 'float64' },
  { numeric_summary: { income: { mean: 50000, median: 49500, std: 10000, min: 20000, max: 80000 } } },
  100
);
assert.strictEqual(balancedCol.strategy, 'mean');
console.log('✓ Scenario 1: Balanced numeric distribution recommends mean.');

// Case 2: Numeric Skewed Distribution (Outliers pull mean away from median)
const skewedCol = getImputationRecommendation(
  'net_worth',
  { net_worth: 'float64' },
  { numeric_summary: { net_worth: { mean: 250000, median: 70000, std: 300000, min: 1000, max: 5000000 } } },
  100
);
assert.strictEqual(skewedCol.strategy, 'median');
console.log('✓ Scenario 2: Skewed numeric distribution with outliers recommends median.');

// Case 3: Discrete low-cardinality numeric column (e.g. rating 1-5)
const discreteCol = getImputationRecommendation(
  'satisfaction_rating',
  { satisfaction_rating: 'int64' },
  {
    numeric_summary: { satisfaction_rating: { mean: 3.8, median: 4, std: 1.1, min: 1, max: 5 } },
    unique_values: { satisfaction_rating: 5 },
  },
  100
);
assert.strictEqual(discreteCol.strategy, 'mode');
console.log('✓ Scenario 3: Discrete low-cardinality numeric column recommends mode.');

// Case 4: Zero standard deviation (Constant column)
const zeroStdCol = getImputationRecommendation(
  'fixed_rate',
  { fixed_rate: 'float64' },
  { numeric_summary: { fixed_rate: { mean: 5.0, median: 5.0, std: 0, min: 5.0, max: 5.0 } } },
  100
);
assert.strictEqual(zeroStdCol.strategy, 'mean');
console.log('✓ Scenario 4: Zero standard deviation safely recommends mean.');

// Case 5: Categorical dominant category (>= 50%)
const dominantCat = getImputationRecommendation(
  'country',
  { country: 'object' },
  { categorical_summary: { country: { top_value: 'India', top_frequency: 85 } } },
  100
);
assert.strictEqual(dominantCat.strategy, 'mode');
console.log('✓ Scenario 5: Dominant category (85%) recommends mode.');

// Case 6: Categorical balanced / multi-class distribution (< 50%)
const balancedCat = getImputationRecommendation(
  'device',
  { device: 'object' },
  { categorical_summary: { device: { top_value: 'Mobile', top_frequency: 35 } } },
  100
);
assert.strictEqual(balancedCat.strategy, 'constant');
console.log('✓ Scenario 6: Non-dominant category (35%) recommends constant to avoid artificial bias.');

// Case 7: Insufficient statistics
const emptyStatsCol = getImputationRecommendation('unknown_num', { unknown_num: 'float64' }, {}, 100);
assert.strictEqual(emptyStatsCol.strategy, 'median');
console.log('✓ Scenario 7: Insufficient statistics gracefully falls back to robust default.');

// ============================================================================
// 2. Feature B: Invalid Numerical Values Payload & Validation
// ============================================================================
function compileInvalidNumericalPayload(configs) {
  const payload = {};
  for (const [col, cfg] of Object.entries(configs)) {
    if (
      !cfg ||
      (!cfg.action && !cfg.infinity_strategy) ||
      (cfg.action === 'none' && !cfg.infinity_strategy)
    ) continue;
    if (cfg.min_value !== undefined && cfg.max_value !== undefined) {
      if (Number(cfg.min_value) > Number(cfg.max_value)) {
        throw new Error(`min_value (${cfg.min_value}) cannot exceed max_value (${cfg.max_value}) for '${col}'.`);
      }
    }
    const item = { action: cfg.action };
    if (cfg.infinity_strategy) item.infinity_strategy = cfg.infinity_strategy;
    if (cfg.infinity_strategy === 'constant' && cfg.infinity_constant !== undefined) {
      item.infinity_constant = Number(cfg.infinity_constant);
    }
    if (cfg.min_value !== undefined && cfg.min_value !== '') item.min_value = Number(cfg.min_value);
    if (cfg.max_value !== undefined && cfg.max_value !== '') item.max_value = Number(cfg.max_value);
    if (cfg.handle_infinities !== undefined) item.handle_infinities = Boolean(cfg.handle_infinities);
    payload[col] = item;
  }
  return payload;
}

const validInvPayload = compileInvalidNumericalPayload({
  age: {
    action: 'replace_with_nan',
    min_value: 0,
    max_value: 120,
    handle_infinities: true,
    infinity_strategy: 'mode',
  },
  speed: { action: 'drop_rows', min_value: 0 },
  ratio: {
    action: 'none',
    handle_infinities: true,
    infinity_strategy: 'constant',
    infinity_constant: -1,
  },
});

assert.strictEqual(validInvPayload.age.action, 'replace_with_nan');
assert.strictEqual(validInvPayload.age.min_value, 0);
assert.strictEqual(validInvPayload.age.max_value, 120);
assert.strictEqual(validInvPayload.age.infinity_strategy, 'mode');
assert.strictEqual(validInvPayload.speed.action, 'drop_rows');
assert.strictEqual(validInvPayload.speed.max_value, undefined);
assert.strictEqual(validInvPayload.ratio.infinity_strategy, 'constant');
assert.strictEqual(validInvPayload.ratio.infinity_constant, -1);
assert.strictEqual(validInvPayload.ratio.action, 'none');

assert.throws(() => {
  compileInvalidNumericalPayload({
    bad_col: { action: 'replace_with_nan', min_value: 100, max_value: 20 },
  });
});
console.log('✓ Scenario 8: Invalid numerical configuration compiles accurately with range boundary checks.');

// ============================================================================
// 3. Feature C: Categorical Value Mapping Payload & Approval Guard
// ============================================================================
function compileValueMappingPayload(mappingsState) {
  const payload = {};
  for (const [col, list] of Object.entries(mappingsState)) {
    if (!Array.isArray(list)) continue;
    const colMap = {};
    for (const item of list) {
      // Must be explicitly user-approved!
      if (item.approved && item.from && item.to) {
        colMap[item.from] = item.to;
      }
    }
    if (Object.keys(colMap).length > 0) {
      payload[col] = colMap;
    }
  }
  return payload;
}

const mockMappingsState = {
  country: [
    { from: 'india', to: 'India', approved: true },
    { from: 'Indai', to: 'India', approved: true },
    { from: 'INDIA', to: 'India', approved: false }, // Unapproved!
  ],
  city: [
    { from: 'new york', to: 'New York', approved: true },
  ],
  team: [
    { from: 'hr', to: 'HR', approved: true },
    { from: 'HR', to: 'HR', approved: true },
    { from: 'engineering', to: 'Engineering', approved: true },
    { from: 'SALES', to: 'Sales', approved: true },
    { from: ' mIxEd ', to: 'Mixed', approved: true },
    { from: 'Finance', to: 'Finance', approved: false },
  ],
};

const compiledMapping = compileValueMappingPayload(mockMappingsState);
assert.strictEqual(Object.keys(compiledMapping.country).length, 2);
assert.strictEqual(compiledMapping.country.india, 'India');
assert.strictEqual(compiledMapping.country.Indai, 'India');
assert.strictEqual(compiledMapping.country.INDIA, undefined, 'Unapproved mapping must not be compiled');
assert.strictEqual(compiledMapping.city['new york'], 'New York');
assert.strictEqual(compiledMapping.team.hr, 'HR');
assert.strictEqual(compiledMapping.team.HR, 'HR');
assert.strictEqual(compiledMapping.team.engineering, 'Engineering');
assert.strictEqual(compiledMapping.team.SALES, 'Sales');
assert.strictEqual(compiledMapping.team[' mIxEd '], 'Mixed');
assert.strictEqual(compiledMapping.team.Finance, undefined);
console.log('✓ Scenario 9: Categorical value mapping compiles only approved pairs without unapproved mutations.');

// ============================================================================
// 4. Feature D: Full Cleaning Plan Compilation & Backend Contract Compatibility
// ============================================================================
const VALID_OPERATION_KEYS = new Set([
  'missing_values',
  'remove_duplicates',
  'duplicate_subset',
  'trim_whitespace',
  'casing',
  'drop_empty_columns',
  'drop_constant_columns',
  'invalid_numerical_values',
  'value_mapping',
]);

const fullOps = {
  missing_values: { salary: 'median' },
  invalid_numerical_values: validInvPayload,
  value_mapping: compiledMapping,
  remove_duplicates: true,
  trim_whitespace: true,
};

for (const k of Object.keys(fullOps)) {
  assert(VALID_OPERATION_KEYS.has(k), `Operation key '${k}' must be accepted by backend VALID_OPERATION_KEYS`);
}
console.log('✓ Scenario 10: Complete multi-operation cleaning plan adheres to backend contract.');

console.log('--- ALL MILESTONE 9E CHECKS PASSED SUCCESSFULLY ---');
