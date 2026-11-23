import React, { useState, useMemo, useEffect, useRef, useCallback } from 'react';
import {
  SlidersHorizontal,
  CheckCircle2,
  AlertTriangle,
  Download,
  Sparkles,
  RefreshCw,
  ArrowRight,
  ShieldCheck,
  FileText,
  Layers,
  Loader2,
  Trash2,
  HelpCircle,
  CaseSensitive,
  Copy,
  Plus,
  ArrowRightLeft,
  Binary,
  Compass,
  RotateCcw,
  Check,
} from 'lucide-react';
import Card from '../common/Card';
import Badge from '../common/Badge';
import Button from '../common/Button';
import DatasetCleaningPlanView from './DatasetCleaningPlanView';
import {
  cleanDataset,
  downloadCleanedDataset,
  getCleanedDatasetPreview,
} from '../../api/datasetApi';
import { getErrorMessage, isDatasetNotFoundError } from '../../api/client';
import { useDataset } from '../../context/DatasetContext';

/**
 * Robust detection of identifier columns (e.g. employee_id, customer_id, uuid).
 * Never recommends statistical mean/median imputation for IDs.
 */
function isIdentifierColumn(col, qualityData, intelligenceData, profileData) {
  if (!col) return false;
  const colLower = col.toLowerCase().trim();

  // 1. Check intelligenceData column_roles
  if (intelligenceData?.column_roles?.[col]) {
    const roleInfo = intelligenceData.column_roles[col];
    if (roleInfo.role === 'potential_id' || roleInfo.is_potential_id) {
      return true;
    }
  }

  // 2. Check qualityData potential_id_columns
  if (Array.isArray(qualityData?.potential_id_columns)) {
    if (qualityData.potential_id_columns.some((item) => item.column === col)) {
      return true;
    }
  }

  // 3. Name heuristic and high uniqueness check
  const idRegex = /(^|_|-)(id|uuid|guid|key|code|pk|identifier)($|_|-)/i;
  const matchesName = idRegex.test(colLower) || colLower.endsWith('id');
  const uniqueCount = profileData?.unique_values?.[col];
  const totalRows = profileData?.total_rows;

  if (matchesName) {
    if (uniqueCount && totalRows && uniqueCount / totalRows > 0.8) {
      return true;
    }
    if (['id', 'uuid', 'guid', 'pk'].includes(colLower) || colLower.endsWith('_id')) {
      return true;
    }
  }

  return false;
}

/**
 * Detects whether a column contains 100% missing values.
 */
function isEmptyColumn(col, qualityData, profileData) {
  if (!col) return false;
  if (Array.isArray(qualityData?.empty_columns)) {
    if (qualityData.empty_columns.some((c) => (typeof c === 'string' ? c === col : c.column === col))) {
      return true;
    }
  }
  if (profileData?.missing_values?.[col]) {
    if (
      profileData.missing_values[col].percentage === 100.0 ||
      (profileData.total_rows > 0 && profileData.missing_values[col].count === profileData.total_rows)
    ) {
      return true;
    }
  }
  return false;
}

function getCanonicalCategoryValue(normalizedValue, variations) {
  const normalized = String(normalizedValue || '').trim();
  if (!normalized) return '';

  const trimmedVariations = variations
    .filter((variation) => typeof variation === 'string')
    .map((variation) => variation.trim());
  const acronym = trimmedVariations.find(
    (variation) =>
      /^[A-Z]{2,4}$/.test(variation) && variation.toLowerCase() === normalized.toLowerCase()
  );
  if (acronym) return acronym;

  const titleCase = trimmedVariations.find(
    (variation) =>
      variation === variation.charAt(0).toUpperCase() + variation.slice(1).toLowerCase() &&
      variation.toLowerCase() === normalized.toLowerCase()
  );
  if (titleCase) return titleCase;

  const preservedCase = trimmedVariations.find(
    (variation) =>
      variation.toLowerCase() === normalized.toLowerCase() &&
      /[a-z]/.test(variation) &&
      /[A-Z]/.test(variation)
  );
  if (preservedCase) return preservedCase;

  return normalized.charAt(0).toUpperCase() + normalized.slice(1);
}

function hasWhitespaceVariations(qualityData) {
  return (qualityData?.inconsistent_categorical_values?.columns || []).some((column) =>
    (column.inconsistencies || []).some((inconsistency) =>
      (inconsistency.variations || []).some(
        (value) => typeof value === 'string' && value !== value.trim()
      )
    )
  );
}

/**
 * Calculates a statistically justified recommendation for handling missing values.
 * Uses distribution shape (skewness, variance, outliers) for numeric columns,
 * and category dominance for categorical columns.
 * Specifically excludes identifiers and completely empty columns from statistical imputation.
 */
function getImputationRecommendation(
  col,
  columnTypes,
  profileData,
  originalDataset,
  qualityData,
  intelligenceData
) {
  const dtype = (columnTypes[col] || '').toLowerCase();
  const isNumeric =
    dtype.includes('int') ||
    dtype.includes('float') ||
    dtype.includes('double') ||
    dtype.includes('number');

  // Check 1: Completely empty columns (100% missing)
  if (isEmptyColumn(col, qualityData, profileData)) {
    return {
      strategy: 'none',
      label: 'Drop Column',
      reason: 'Recommended: Drop Column — Column contains 100% missing values (0 observations available to calculate statistical estimates). Drop this column in Column Pruning below.',
      isIdentifier: false,
      isEmpty: true,
      stats: '100% missing values',
    };
  }

  // Check 2: Identifier columns (employee_id, user_id, uuid)
  if (isIdentifierColumn(col, qualityData, intelligenceData, profileData)) {
    return {
      strategy: 'none',
      label: 'Leave Unchanged',
      reason: 'Recommended: Leave Unchanged — Identified as an entity identifier column (ID/key). Imputing averages or modes into identifiers can corrupt record identity.',
      isIdentifier: true,
      isEmpty: false,
      stats: 'Entity identifier / natural key',
    };
  }

  if (isNumeric) {
    const numStats = profileData?.numeric_summary?.[col];
    const uniqueCount = profileData?.unique_values?.[col];

    if (!numStats || numStats.mean === null || numStats.median === null) {
      return {
        strategy: 'median',
        label: 'Median',
        reason: 'Recommended: Median — Standard robust default when detailed distribution statistics are limited.',
        isDefaultFallback: true,
        stats: 'Limited distribution stats',
      };
    }

    const { mean, median, std, min, max } = numStats;

    // Rule 1: Discrete integer scale with very few unique values (<= 5)
    if (uniqueCount && uniqueCount <= 5 && Number.isInteger(min) && Number.isInteger(max)) {
      return {
        strategy: 'mode',
        label: 'Mode',
        reason: `Recommended: Mode — Discrete scale with only ${uniqueCount} unique values. Preserves discrete steps without introducing fractional averages.`,
        stats: `Distinct values: ${uniqueCount}, range: [${min}, ${max}]`,
      };
    }

    // Rule 2: Constant / Zero variation
    if (std === 0 || min === max) {
      return {
        strategy: 'mean',
        label: `Mean (${mean})`,
        reason: `Recommended: Mean (${mean}) — Zero variation detected (all non-null observations are ${mean}).`,
        stats: `Mean: ${mean}, Std: 0`,
      };
    }

    // Rule 3: Skewness / Outliers evaluation (|mean - median| / std > 0.2)
    const skewnessRatio = std > 0 ? Math.abs(mean - median) / std : 0;
    if (skewnessRatio > 0.2) {
      return {
        strategy: 'median',
        label: `Median (${median})`,
        reason: `Recommended: Median (${median}) — Distribution is skewed by outliers (|mean - median| = ${Math.abs(mean - median).toFixed(2)}, std = ${std}). Median is more resistant to extreme values.`,
        stats: `Mean: ${mean}, Median: ${median}, Std: ${std}, Min: ${min}, Max: ${max}`,
      };
    }

    // Rule 4: Symmetric distribution without strong skew
    return {
      strategy: 'mean',
      label: `Mean (${mean})`,
      reason: `Recommended: Mean (${mean}) — Distribution is approximately symmetric (mean ${mean} ≈ median ${median}). Mean preserves mathematical sum.`,
      stats: `Mean: ${mean}, Median: ${median}, Std: ${std}`,
    };
  } else {
    // Categorical column
    const catStats = profileData?.categorical_summary?.[col];
    const totalRows = profileData?.total_rows || originalDataset?.row_count;

    if (!catStats || !catStats.top_value || !totalRows) {
      return {
        strategy: 'constant',
        defaultConstant: 'Unknown',
        label: "Custom Constant ('Unknown')",
        reason: "Recommended: Custom Constant ('Unknown') — Insufficient frequency data to confirm a dominant category.",
        stats: 'Limited category stats',
      };
    }

    const { top_value, top_frequency } = catStats;
    const dominancePct = Math.round((top_frequency / totalRows) * 100);

    // Rule: Dominant category (>= 50% of records)
    if (dominancePct >= 50) {
      return {
        strategy: 'mode',
        label: `Mode ('${top_value}')`,
        reason: `Recommended: Mode ('${top_value}') — Dominant category representing ${dominancePct}% of all records (${top_frequency}/${totalRows}).`,
        stats: `Top category: "${top_value}" (${dominancePct}%)`,
      };
    }

    // Balanced / multi-class distribution: mode would artificially overrepresent one category
    return {
      strategy: 'constant',
      defaultConstant: 'Unknown',
      label: "Custom Constant ('Unknown')",
      reason: `Recommended: Custom Constant ('Unknown') — No single dominant class (top category '${top_value}' accounts for only ${dominancePct}%). Imputing the mode would skew category balance.`,
      stats: `Top category: "${top_value}" (${dominancePct}%)`,
    };
  }
}

export default function DatasetCleaningView({
  originalDataset,
  qualityData,
  profileData,
  intelligenceData,
}) {
  const {
    activeDataset,
    cleanedDataset,
    activeVersion,
    isCleanedActive,
    setCleanedDataset: saveCleanedToContext,
    switchDatasetVersion,
    handleDatasetExpiredOrMissing,
  } = useDataset();

  // Primary dataset ID to clean is always the original dataset
  const targetDatasetId = originalDataset?.dataset_id || activeDataset?.dataset_id;

  // Extract column information and data types
  const columnsList = useMemo(() => {
    if (profileData?.columns && Array.isArray(profileData.columns)) {
      return profileData.columns;
    }
    if (originalDataset?.columns && Array.isArray(originalDataset.columns)) {
      return originalDataset.columns;
    }
    return [];
  }, [profileData, originalDataset]);

  const columnTypes = useMemo(() => {
    if (profileData?.column_types && typeof profileData.column_types === 'object') {
      return profileData.column_types;
    }
    return {};
  }, [profileData]);

  // Helper to determine if a column is numeric
  const isNumericColumn = useCallback(
    (col) => {
      const dtype = (columnTypes[col] || '').toLowerCase();
      return (
        dtype.includes('int') ||
        dtype.includes('float') ||
        dtype.includes('double') ||
        dtype.includes('number')
      );
    },
    [columnTypes]
  );

  // Helper to determine if a column is string/text
  const isTextColumn = useCallback(
    (col) => {
      const dtype = (columnTypes[col] || '').toLowerCase();
      if (!dtype) return true;
      return (
        dtype.includes('object') ||
        dtype.includes('str') ||
        dtype.includes('char') ||
        dtype.includes('text') ||
        dtype.includes('category')
      );
    },
    [columnTypes]
  );

  // Columns with missing values
  const columnsWithMissingValues = useMemo(() => {
    const cols = [];
    if (qualityData?.missing_values?.columns) {
      qualityData.missing_values.columns.forEach((c) => {
        if (!cols.includes(c.column)) cols.push(c.column);
      });
    }
    if (profileData?.missing_values) {
      Object.entries(profileData.missing_values).forEach(([col, info]) => {
        if (info.count > 0 && !cols.includes(col)) cols.push(col);
      });
    }
    return cols;
  }, [qualityData, profileData]);

  // Columns with invalid numerical values (+inf / -inf)
  const columnsWithInfinities = useMemo(() => {
    const cols = [];
    if (qualityData?.invalid_numerical_values?.columns) {
      qualityData.invalid_numerical_values.columns.forEach((c) => {
        cols.push({
          column: c.column,
          total: c.total_invalid_count,
          pos_inf: c.positive_infinity_count,
          neg_inf: c.negative_infinity_count,
          finite_value_count: c.finite_value_count,
          mode_available: c.mode_available,
          mode_unavailable_reason: c.mode_unavailable_reason,
        });
      });
    }
    return cols;
  }, [qualityData]);

  // Form Configuration State
  const [missingConfigs, setMissingConfigs] = useState({});
  const [invalidNumConfigs, setInvalidNumConfigs] = useState({});
  const [valueMappings, setValueMappings] = useState({});
  const [removeDuplicates, setRemoveDuplicates] = useState(false);
  const [useDuplicateSubset, setUseDuplicateSubset] = useState(false);
  const [duplicateSubset, setDuplicateSubset] = useState([]);
  const [trimAllWhitespace, setTrimAllWhitespace] = useState(false);
  const [selectedTrimColumns, setSelectedTrimColumns] = useState([]);
  const [casingConfigs, setCasingConfigs] = useState({});
  const [dropEmptyColumns, setDropEmptyColumns] = useState(false);
  const [dropConstantColumns, setDropConstantColumns] = useState(false);

  // Review & Approval State
  const [showApprovalModal, setShowApprovalModal] = useState(false);
  const [isCleaning, setIsCleaning] = useState(false);
  const [cleaningError, setCleaningError] = useState(null);

  // Cleaned Results State
  const [cleaningResult, setCleaningResult] = useState(() => {
    if (cleanedDataset?.report) {
      return {
        cleaned_dataset_id: cleanedDataset.dataset_id,
        filename: cleanedDataset.filename,
        download_url: cleanedDataset.download_url,
        data: cleanedDataset.report,
      };
    }
    return null;
  });

  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState(null);
  const [cleanedPreview, setCleanedPreview] = useState(null);
  const [isPreviewLoading, setIsPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState(null);

  // Compute all recommendations function
  const computeAllRecommendations = useCallback(() => {
    // 1. Missing values recommendations
    const recMissing = {};
    columnsList.forEach((col) => {
      const isAffected = columnsWithMissingValues.includes(col);
      if (isAffected) {
        const rec = getImputationRecommendation(
          col,
          columnTypes,
          profileData,
          originalDataset,
          qualityData,
          intelligenceData
        );
        if (rec.strategy && rec.strategy !== 'none') {
          recMissing[col] = {
            strategy: rec.strategy,
            value: rec.defaultConstant || '',
          };
        } else {
          recMissing[col] = {
            strategy: 'none',
            value: '',
          };
        }
      }
    });

    // 2. Invalid numerical values recommendations (infinities)
    const recInvalidNum = {};
    columnsWithInfinities.forEach((infObj) => {
      recInvalidNum[infObj.column] = {
        action: 'none',
        infinity_strategy: infObj.mode_available
          ? 'mode'
          : infObj.finite_value_count > 0
            ? 'median'
            : 'leave_unchanged',
        min_value: '',
        max_value: '',
        handle_infinities: true,
      };
    });

    // 3. Categorical value mapping (preselect detected casing/whitespace variations)
    const recValueMappings = {};
    if (qualityData?.inconsistent_categorical_values?.columns) {
      qualityData.inconsistent_categorical_values.columns.forEach((colObj) => {
        const col = colObj.column;
        if (colObj.inconsistencies) {
          const list = [];
          colObj.inconsistencies.forEach((inc) => {
            const rawVariations = inc.variations || [];
            const normKey = inc.normalized_value || '';
            const suggestedCanonical = getCanonicalCategoryValue(normKey, rawVariations);

            rawVariations.forEach((variant) => {
              list.push({
                from: variant,
                to: suggestedCanonical,
                approved: true, // Preselected for user review
                isSuggested: true,
              });
            });
          });
          if (list.length > 0) {
            recValueMappings[col] = list;
          }
        }
      });
    }

    // 4. Other operations recommendations
    const recRemoveDuplicates = Boolean(qualityData?.duplicate_rows?.duplicate_count > 0);
    const recDropEmpty = Boolean(qualityData?.empty_columns?.length > 0);
    const recDropConstant = Boolean(qualityData?.constant_columns?.length > 0);
    const recTrimWhitespace = hasWhitespaceVariations(qualityData);

    return {
      missingConfigs: recMissing,
      invalidNumConfigs: recInvalidNum,
      valueMappings: recValueMappings,
      removeDuplicates: recRemoveDuplicates,
      dropEmptyColumns: recDropEmpty,
      dropConstantColumns: recDropConstant,
      trimAllWhitespace: recTrimWhitespace,
    };
  }, [
    columnsList,
    columnsWithMissingValues,
    columnsWithInfinities,
    columnTypes,
    profileData,
    originalDataset,
    qualityData,
    intelligenceData,
  ]);

  // Track initialization per dataset to preselect recommendations by default
  const initializedDatasetRef = useRef(null);
  const initializedMappingsDatasetRef = useRef(null);

  useEffect(() => {
    if (!targetDatasetId) return;

    // When dataset or metadata loads, preselect recommendations by default
    const profileIsReady =
      Array.isArray(profileData?.columns) &&
      profileData.columns.length > 0 &&
      profileData?.column_types &&
      profileData.columns.every((col) => profileData.column_types[col] !== undefined);
    if (initializedDatasetRef.current !== targetDatasetId && profileIsReady && qualityData) {
      const recs = computeAllRecommendations();
      setMissingConfigs(recs.missingConfigs);
      setInvalidNumConfigs(recs.invalidNumConfigs);
      setValueMappings(qualityData ? recs.valueMappings : {});
      setRemoveDuplicates(false);
      setDropEmptyColumns(false);
      setDropConstantColumns(false);
      setTrimAllWhitespace(recs.trimAllWhitespace);
      initializedDatasetRef.current = targetDatasetId;
      if (qualityData) {
        initializedMappingsDatasetRef.current = targetDatasetId;
      }
    } else if (
      profileIsReady &&
      qualityData &&
      initializedMappingsDatasetRef.current !== targetDatasetId
    ) {
      const recs = computeAllRecommendations();
      setValueMappings((prev) => {
        const mappings = { ...recs.valueMappings };
        Object.entries(prev).forEach(([col, list]) => {
          const customMappings = list.filter((item) => item.isCustom);
          if (customMappings.length > 0) {
            mappings[col] = [...(mappings[col] || []), ...customMappings];
          }
        });
        return mappings;
      });
      initializedMappingsDatasetRef.current = targetDatasetId;
    }
  }, [targetDatasetId, profileData, qualityData, computeAllRecommendations]);

  // Explicit user action to re-apply all recommendations
  const handleApplyAllRecommendations = () => {
    const recs = computeAllRecommendations();
    setMissingConfigs(recs.missingConfigs);
    setInvalidNumConfigs(recs.invalidNumConfigs);
    setValueMappings(recs.valueMappings);
    setRemoveDuplicates(recs.removeDuplicates);
    setDropEmptyColumns(recs.dropEmptyColumns);
    setDropConstantColumns(recs.dropConstantColumns);
    setTrimAllWhitespace(recs.trimAllWhitespace);
  };

  // Reset all operations to Leave Unchanged
  const handleResetToUnchanged = () => {
    const clearedMissing = {};
    columnsList.forEach((c) => {
      clearedMissing[c] = { strategy: 'none', value: '' };
    });
    setMissingConfigs(clearedMissing);
    setInvalidNumConfigs({});
    setValueMappings((prev) => {
      const updated = {};
      Object.entries(prev).forEach(([col, list]) => {
        updated[col] = list.map((item) => ({ ...item, approved: false }));
      });
      return updated;
    });
    setRemoveDuplicates(false);
    setUseDuplicateSubset(false);
    setDuplicateSubset([]);
    setTrimAllWhitespace(false);
    setSelectedTrimColumns([]);
    setCasingConfigs({});
    setDropEmptyColumns(false);
    setDropConstantColumns(false);
  };

  // Compile full operations payload matching verified backend contract
  const compiledOperations = useMemo(() => {
    const ops = {};

    // 1. Missing Values
    const activeMissing = {};
    Object.entries(missingConfigs).forEach(([col, conf]) => {
      if (!conf || !conf.strategy || conf.strategy === 'none') return;
      if (isEmptyColumn(col, qualityData, profileData)) return;
      if (conf.strategy === 'constant') {
        activeMissing[col] = {
          strategy: 'constant',
          value: conf.value !== undefined ? conf.value : '',
        };
      } else {
        activeMissing[col] = conf.strategy;
      }
    });
    if (Object.keys(activeMissing).length > 0) {
      ops.missing_values = activeMissing;
    }

    // 2. Invalid Numerical Values
    const activeInvalidNum = {};
    Object.entries(invalidNumConfigs).forEach(([col, conf]) => {
      if (
        !conf ||
        (!conf.action && !conf.infinity_strategy) ||
        (conf.action === 'none' && !conf.infinity_strategy)
      ) return;
      const colPayload = { action: conf.action || 'none' };
      if (conf.infinity_strategy) {
        colPayload.infinity_strategy = conf.infinity_strategy;
      }
      if (conf.infinity_strategy === 'constant' && conf.infinity_constant !== '') {
        colPayload.infinity_constant = Number(conf.infinity_constant);
      }
      if (conf.min_value !== undefined && conf.min_value !== '') {
        colPayload.min_value = Number(conf.min_value);
      }
      if (conf.max_value !== undefined && conf.max_value !== '') {
        colPayload.max_value = Number(conf.max_value);
      }
      if (conf.handle_infinities !== undefined) {
        colPayload.handle_infinities = Boolean(conf.handle_infinities);
      }
      activeInvalidNum[col] = colPayload;
    });
    if (Object.keys(activeInvalidNum).length > 0) {
      ops.invalid_numerical_values = activeInvalidNum;
    }

    // 3. Categorical Value Mapping (approved pairs only)
    const activeMapping = {};
    Object.entries(valueMappings).forEach(([col, list]) => {
      if (!Array.isArray(list)) return;
      const approvedMap = {};
      list.forEach((item) => {
        if (item.approved && item.from !== undefined && item.from !== '' && item.to !== undefined) {
          approvedMap[item.from] = item.to;
        }
      });
      if (Object.keys(approvedMap).length > 0) {
        activeMapping[col] = approvedMap;
      }
    });
    if (Object.keys(activeMapping).length > 0) {
      ops.value_mapping = activeMapping;
    }

    // 4. Duplicates
    if (removeDuplicates) {
      ops.remove_duplicates = true;
      if (useDuplicateSubset && duplicateSubset.length > 0) {
        ops.duplicate_subset = duplicateSubset;
      }
    }

    // 5. Text Normalization
    if (trimAllWhitespace) {
      ops.trim_whitespace = true;
    } else if (selectedTrimColumns.length > 0) {
      ops.trim_whitespace = selectedTrimColumns;
    }

    const activeCasing = {};
    Object.entries(casingConfigs).forEach(([col, targetCase]) => {
      if (targetCase && targetCase !== 'none') {
        activeCasing[col] = targetCase;
      }
    });
    if (Object.keys(activeCasing).length > 0) {
      ops.casing = activeCasing;
    }

    // 6. Empty & Constant Columns
    if (dropEmptyColumns) {
      ops.drop_empty_columns = true;
    }
    if (dropConstantColumns) {
      ops.drop_constant_columns = true;
    }

    return ops;
  }, [
    missingConfigs,
    invalidNumConfigs,
    valueMappings,
    removeDuplicates,
    useDuplicateSubset,
    duplicateSubset,
    trimAllWhitespace,
    selectedTrimColumns,
    casingConfigs,
    dropEmptyColumns,
    dropConstantColumns,
    qualityData,
    profileData,
  ]);

  const hasSelectedOperations = useMemo(() => {
    return Object.keys(compiledOperations).length > 0;
  }, [compiledOperations]);

  // Check if any operation deletes rows
  const willDeleteRows = useMemo(() => {
    if (compiledOperations.remove_duplicates) return true;
    if (compiledOperations.invalid_numerical_values) {
      return Object.values(compiledOperations.invalid_numerical_values).some(
        (cfg) => cfg.action === 'drop_rows'
      );
    }
    return false;
  }, [compiledOperations]);

  // Count operations chosen
  const operationsCountSummary = useMemo(() => {
    let count = 0;
    if (compiledOperations.missing_values) {
      count += Object.keys(compiledOperations.missing_values).length;
    }
    if (compiledOperations.invalid_numerical_values) {
      count += Object.keys(compiledOperations.invalid_numerical_values).length;
    }
    if (compiledOperations.value_mapping) {
      count += Object.keys(compiledOperations.value_mapping).length;
    }
    if (compiledOperations.remove_duplicates) count += 1;
    if (compiledOperations.trim_whitespace) count += 1;
    if (compiledOperations.casing) {
      count += Object.keys(compiledOperations.casing).length;
    }
    if (compiledOperations.drop_empty_columns) count += 1;
    if (compiledOperations.drop_constant_columns) count += 1;
    return count;
  }, [compiledOperations]);
  const selectedImputationCount = compiledOperations.missing_values
    ? Object.keys(compiledOperations.missing_values).length
    : 0;

  // Execute cleaning upon explicit user confirmation
  const handleExecuteClean = async () => {
    if (!targetDatasetId || !hasSelectedOperations) return;

    setIsCleaning(true);
    setCleaningError(null);
    setShowApprovalModal(false);

    try {
      const response = await cleanDataset(targetDatasetId, compiledOperations);

      saveCleanedToContext({
        dataset_id: response.cleaned_dataset_id,
        filename: response.filename,
        row_count: response.data?.cleaned_shape?.rows,
        column_count: response.data?.cleaned_shape?.columns,
        download_url: response.download_url,
        report: response.data,
      });

      setCleanedPreview(null);
      setPreviewError(null);
      setCleaningResult(response);
    } catch (err) {
      if (isDatasetNotFoundError(err)) {
        handleDatasetExpiredOrMissing();
        return;
      }
      setCleaningError(getErrorMessage(err, 'Failed to clean dataset. Please check parameters.'));
    } finally {
      setIsCleaning(false);
    }
  };

  const handleDownload = async () => {
    const idToDownload = cleaningResult?.cleaned_dataset_id || cleanedDataset?.dataset_id;
    if (!idToDownload) return;

    setIsDownloading(true);
    setDownloadError(null);
    try {
      const filename =
        cleaningResult?.filename || cleanedDataset?.filename || 'cleaned_dataset.csv';
      await downloadCleanedDataset(idToDownload, filename);
    } catch (err) {
      setDownloadError(getErrorMessage(err, 'Failed to download cleaned CSV.'));
    } finally {
      setIsDownloading(false);
    }
  };

  const handleViewCleanedData = async () => {
    const idToPreview = cleaningResult?.cleaned_dataset_id || cleanedDataset?.dataset_id;
    if (!idToPreview) return;

    setIsPreviewLoading(true);
    setPreviewError(null);
    try {
      const preview = await getCleanedDatasetPreview(idToPreview);
      setCleanedPreview(preview);
    } catch (err) {
      setPreviewError(getErrorMessage(err, 'Failed to preview the cleaned CSV.'));
    } finally {
      setIsPreviewLoading(false);
    }
  };

  const handleClosePreview = () => {
    setCleanedPreview(null);
    setPreviewError(null);
  };

  const getRecommendation = (col) =>
    getImputationRecommendation(
      col,
      columnTypes,
      profileData,
      originalDataset,
      qualityData,
      intelligenceData
    );
  const hasDetectedWhitespaceVariations = hasWhitespaceVariations(qualityData);

  return (
    <DatasetCleaningPlanView
      originalDataset={originalDataset}
      cleanedDataset={cleanedDataset}
      isCleanedActive={isCleanedActive}
      switchDatasetVersion={switchDatasetVersion}
      profileData={profileData}
      qualityData={qualityData}
      columnsList={columnsList}
      columnTypes={columnTypes}
      columnsWithMissingValues={columnsWithMissingValues}
      columnsWithInfinities={columnsWithInfinities}
      isNumericColumn={isNumericColumn}
      isTextColumn={isTextColumn}
      getRecommendation={getRecommendation}
      missingConfigs={missingConfigs}
      setMissingConfigs={setMissingConfigs}
      invalidNumConfigs={invalidNumConfigs}
      setInvalidNumConfigs={setInvalidNumConfigs}
      valueMappings={valueMappings}
      setValueMappings={setValueMappings}
      removeDuplicates={removeDuplicates}
      setRemoveDuplicates={setRemoveDuplicates}
      useDuplicateSubset={useDuplicateSubset}
      setUseDuplicateSubset={setUseDuplicateSubset}
      duplicateSubset={duplicateSubset}
      setDuplicateSubset={setDuplicateSubset}
      trimAllWhitespace={trimAllWhitespace}
      setTrimAllWhitespace={setTrimAllWhitespace}
      hasWhitespaceVariations={hasDetectedWhitespaceVariations}
      casingConfigs={casingConfigs}
      setCasingConfigs={setCasingConfigs}
      dropEmptyColumns={dropEmptyColumns}
      setDropEmptyColumns={setDropEmptyColumns}
      dropConstantColumns={dropConstantColumns}
      setDropConstantColumns={setDropConstantColumns}
      compiledOperations={compiledOperations}
      hasSelectedOperations={hasSelectedOperations}
      willDeleteRows={willDeleteRows}
      dataReady={Boolean(profileData && qualityData)}
      cleaningError={cleaningError}
      downloadError={downloadError}
      cleaningResult={cleaningResult}
      isCleaning={isCleaning}
      isDownloading={isDownloading}
      cleanedPreview={cleanedPreview}
      isPreviewLoading={isPreviewLoading}
      previewError={previewError}
      showApprovalModal={showApprovalModal}
      setShowApprovalModal={setShowApprovalModal}
      onExecuteClean={handleExecuteClean}
      onDownload={handleDownload}
      onViewCleanedData={handleViewCleanedData}
      onClosePreview={handleClosePreview}
    />
  );

  return (
    <div className="space-y-6">
      {/* Editorial Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 pb-4 border-b border-stone">
        <div>
          <h2 className="font-serif text-2xl font-bold text-navy flex items-center gap-2.5">
            <SlidersHorizontal className="w-6 h-6 text-teal" />
            Clean & Transform
          </h2>
          <p className="text-xs text-navy/70 font-sans mt-1">
            Recommendation-first data intelligence. Review suggested transformations, customize as needed, and approve when ready.
          </p>
        </div>

        {/* Dataset Version Indicator */}
        <div className="flex items-center gap-2">
          {cleanedDataset ? (
            <div className="flex items-center gap-2 bg-white px-3 py-1.5 border border-stone rounded-sm text-xs">
              <span className="text-navy/50 font-mono text-[11px]">ACTIVE VIEW:</span>
              {isCleanedActive ? (
                <Badge variant="teal">Cleaned Version</Badge>
              ) : (
                <Badge variant="stone">Original Version</Badge>
              )}
              <button
                onClick={() => switchDatasetVersion(isCleanedActive ? 'original' : 'cleaned')}
                className="text-[11px] text-teal hover:underline font-semibold ml-1 flex items-center gap-1"
              >
                <ArrowRightLeft className="w-3 h-3" />
                Switch to {isCleanedActive ? 'Original' : 'Cleaned'}
              </button>
            </div>
          ) : (
            <Badge variant="stone">Original Dataset (Cleaned version not yet generated)</Badge>
          )}
        </div>
      </div>

      {/* CORE RECOMMENDATION-FIRST HERO BANNER */}
      <div className="p-5 bg-white border border-teal/30 rounded-sm shadow-xs space-y-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-start gap-3">
            <div className="p-2 bg-sage/40 rounded border border-teal/20 text-teal">
              <Sparkles className="w-5 h-5 text-teal" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="font-serif text-lg font-bold text-navy">
                  Intelligent Cleaning Plan (Recommended)
                </h3>
                <Badge variant="teal" size="sm">
                  Recommendation-First
                </Badge>
              </div>
              <p className="text-xs text-navy/70 font-sans mt-0.5">
                ModelMind analyzed column distributions, anomalies, and identifier roles. Safe recommendations are preselected below.
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            <Button
              variant="outline"
              size="sm"
              icon={RotateCcw}
              onClick={handleResetToUnchanged}
            >
              Reset to Unchanged
            </Button>
            <Button
              variant="primary"
              size="sm"
              icon={Sparkles}
              onClick={handleApplyAllRecommendations}
            >
              Apply All Recommendations
            </Button>
          </div>
        </div>

        {/* Quick Summary Chips of Recommendations and Selected Imputations */}
        <div className="flex flex-wrap items-center gap-2 pt-2 border-t border-stone/60 text-xs font-sans">
          <span className="text-[11px] text-navy/50 font-mono uppercase">Recommendations:</span>

          {selectedImputationCount > 0 ? (
            <span className="inline-flex items-center gap-1 px-2.5 py-1 bg-sage/30 text-teal-dark border border-teal/20 rounded text-[11px] font-medium">
              <Check className="w-3 h-3" />
              {selectedImputationCount} Missing Value Imputation(s) Selected
            </span>
          ) : (
            <span className="text-[11px] text-navy/50 italic">
              {columnsWithMissingValues.length > 0
                ? 'No missing-value imputations selected'
                : 'No missing values'}
            </span>
          )}

          {columnsWithInfinities.length > 0 && (
            <span className="inline-flex items-center gap-1 px-2.5 py-1 bg-red-50 text-red-700 border border-red-200 rounded text-[11px] font-medium">
              <Check className="w-3 h-3" />
              {columnsWithInfinities.length} Infinite Numbers Handled
            </span>
          )}

          {qualityData?.inconsistent_categorical_values?.total_affected_columns > 0 && (
            <span className="inline-flex items-center gap-1 px-2.5 py-1 bg-stone-light/60 text-navy/80 border border-stone rounded text-[11px] font-medium">
              <Check className="w-3 h-3" />
              Categorical Casing & Whitespace Mappings Preselected
            </span>
          )}

          {qualityData?.duplicate_rows?.duplicate_count > 0 && (
            <span className="inline-flex items-center gap-1 px-2.5 py-1 bg-amber-50 text-amber-800 border border-amber-200 rounded text-[11px] font-medium">
              <Check className="w-3 h-3" />
              Deduplication Recommended ({qualityData.duplicate_rows.duplicate_count} rows)
            </span>
          )}

          {qualityData?.empty_columns?.length > 0 && (
            <span className="inline-flex items-center gap-1 px-2.5 py-1 bg-red-50 text-red-700 border border-red-200 rounded text-[11px] font-medium">
              <Check className="w-3 h-3" />
              Drop {qualityData.empty_columns.length} Empty Column(s)
            </span>
          )}

          {qualityData?.constant_columns?.length > 0 && (
            <span className="inline-flex items-center gap-1 px-2.5 py-1 bg-amber-50 text-amber-800 border border-amber-200 rounded text-[11px] font-medium">
              <Check className="w-3 h-3" />
              Drop {qualityData.constant_columns.length} Constant Column(s)
            </span>
          )}
        </div>
      </div>

      {/* Cleaned Dataset Summary (When cleaning has been executed) */}
      {cleaningResult && (
        <Card
          title="Cleaned Dataset Summary"
          subtitle={`Generated file: ${cleaningResult.filename || 'cleaned_dataset.csv'}`}
          headerAction={
            <div className="flex items-center gap-2">
              <Button
                variant="primary"
                size="sm"
                icon={Download}
                onClick={handleDownload}
                loading={isDownloading}
              >
                Download Cleaned CSV
              </Button>
              {!isCleanedActive && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => switchDatasetVersion('cleaned')}
                >
                  Set as Active Dataset
                </Button>
              )}
            </div>
          }
        >
          <div className="space-y-4 text-xs font-mono">
            {/* Shape Comparison */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-sans">
              <div className="p-3 bg-white border border-stone rounded-sm">
                <span className="text-navy/50 text-[10px] block uppercase font-mono mb-1">ROWS</span>
                <div className="flex items-center gap-2">
                  <span className="text-navy font-semibold text-base">
                    {cleaningResult.data?.cleaned_shape?.rows?.toLocaleString()}
                  </span>
                  {cleaningResult.data?.initial_shape?.rows !== cleaningResult.data?.cleaned_shape?.rows && (
                    <Badge variant="amber" size="sm">
                      {cleaningResult.data?.cleaned_shape?.rows - cleaningResult.data?.initial_shape?.rows} rows
                    </Badge>
                  )}
                </div>
                <span className="text-[11px] text-navy/40 block mt-0.5">
                  Initial: {cleaningResult.data?.initial_shape?.rows?.toLocaleString()}
                </span>
              </div>

              <div className="p-3 bg-white border border-stone rounded-sm">
                <span className="text-navy/50 text-[10px] block uppercase font-mono mb-1">COLUMNS</span>
                <div className="flex items-center gap-2">
                  <span className="text-navy font-semibold text-base">
                    {cleaningResult.data?.cleaned_shape?.columns}
                  </span>
                  {cleaningResult.data?.initial_shape?.columns !== cleaningResult.data?.cleaned_shape?.columns && (
                    <Badge variant="amber" size="sm">
                      {cleaningResult.data?.cleaned_shape?.columns - cleaningResult.data?.initial_shape?.columns} cols
                    </Badge>
                  )}
                </div>
                <span className="text-[11px] text-navy/40 block mt-0.5">
                  Initial: {cleaningResult.data?.initial_shape?.columns}
                </span>
              </div>

              <div className="p-3 bg-white border border-stone rounded-sm">
                <span className="text-navy/50 text-[10px] block uppercase font-mono mb-1">OPERATIONS</span>
                <span className="text-teal font-semibold text-base block">
                  {cleaningResult.data?.operations_count || cleaningResult.data?.operations_performed?.length || 0}
                </span>
                <span className="text-[11px] text-navy/40 block mt-0.5">Directives applied</span>
              </div>

              <div className="p-3 bg-white border border-stone rounded-sm">
                <span className="text-navy/50 text-[10px] block uppercase font-mono mb-1">DATASET ID</span>
                <span className="text-navy font-mono text-[11px] truncate block select-all">
                  {cleaningResult.cleaned_dataset_id}
                </span>
                <span className="text-[11px] text-teal font-sans block mt-0.5">Ready for modeling</span>
              </div>
            </div>

            {/* List of Operations Executed */}
            {cleaningResult.data?.operations_performed && cleaningResult.data.operations_performed.length > 0 && (
              <div className="space-y-2 font-sans">
                <span className="font-semibold text-navy text-xs block">Operations Executed:</span>
                <div className="space-y-1.5 font-mono text-xs">
                  {cleaningResult.data.operations_performed.map((op, idx) => (
                    <div
                      key={idx}
                      className="p-2.5 bg-sage/20 border border-teal/20 rounded-sm flex items-start justify-between gap-3"
                    >
                      <div className="flex items-start gap-2">
                        <CheckCircle2 className="w-4 h-4 text-teal shrink-0 mt-0.5" />
                        <span className="text-navy font-medium">{op.description}</span>
                      </div>
                      {op.affected_rows !== undefined && (
                        <Badge variant="teal" size="sm">
                          {op.affected_rows} row(s)
                        </Badge>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </Card>
      )}

      {/* SECTION 1: Missing Values Imputation & Recommendations */}
      <Card
        title="1. Missing Values Imputation"
        subtitle="Intelligent statistical recommendations are preselected by default. Identifier columns are protected from statistical averages."
        headerAction={
          <Button
            variant="outline"
            size="sm"
            icon={Sparkles}
            onClick={() => {
              const recs = computeAllRecommendations();
              setMissingConfigs(recs.missingConfigs);
            }}
          >
            Apply Imputation Recommendations
          </Button>
        }
      >
        <div className="space-y-4 text-xs font-sans">
          <p className="text-navy/70">
            Columns containing missing values are preconfigured with their recommended strategy based on distribution symmetry, outliers, and category dominance. You can customize any column or select Leave Unchanged.
          </p>

          <div className="border border-stone rounded-sm overflow-hidden">
            <table className="w-full text-left font-mono text-xs">
              <thead className="bg-stone-light/50 border-b border-stone text-navy/70 text-[11px]">
                <tr>
                  <th className="p-2.5">COLUMN</th>
                  <th className="p-2.5">TYPE & ROLE</th>
                  <th className="p-2.5">INTELLIGENT RECOMMENDATION</th>
                  <th className="p-2.5">CURRENT SELECTION</th>
                  <th className="p-2.5">CUSTOM VALUE</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-stone">
                {columnsList.map((col) => {
                  const isNumeric = isNumericColumn(col);
                  const isAffected = columnsWithMissingValues.includes(col);
                  const isId = isIdentifierColumn(col, qualityData, intelligenceData, profileData);
                  const isEmpty = isEmptyColumn(col, qualityData, profileData);
                  const currentConf = isEmpty
                    ? { strategy: 'none', value: '' }
                    : missingConfigs[col] || { strategy: 'none', value: '' };

                  const rec = isAffected || isEmpty
                    ? getImputationRecommendation(
                        col,
                        columnTypes,
                        profileData,
                        originalDataset,
                        qualityData,
                        intelligenceData
                      )
                    : {
                        strategy: 'none',
                        label: 'No imputation needed',
                        reason: 'No missing values detected; no imputation is recommended.',
                      };

                  // Determine status badge
                  const isRecSelected = currentConf.strategy === rec.strategy;
                  const isNone = currentConf.strategy === 'none';

                  return (
                    <tr
                      key={col}
                      className={`hover:bg-stone-light/20 ${isAffected ? 'bg-amber-50/15' : ''}`}
                    >
                      {/* Column Name */}
                      <td className="p-2.5 font-semibold text-navy">
                        <div className="flex items-center gap-1.5">
                          <span>{col}</span>
                          {isAffected && (
                            <Badge variant="amber" size="sm">
                              Missing
                            </Badge>
                          )}
                          {isId && (
                            <Badge variant="stone" size="sm">
                              ID Key
                            </Badge>
                          )}
                          {isEmpty && (
                            <Badge variant="red" size="sm">
                              100% Empty
                            </Badge>
                          )}
                        </div>
                      </td>

                      {/* Type */}
                      <td className="p-2.5">
                        <Badge variant={isNumeric ? 'teal' : 'stone'} size="sm">
                          {columnTypes[col] || (isNumeric ? 'numeric' : 'text')}
                        </Badge>
                      </td>

                      {/* Recommendation & Statistical Evidence */}
                      <td className="p-2.5 font-sans max-w-sm">
                        <div className="space-y-1">
                          <div className="flex items-center gap-1.5">
                            <span className="font-semibold text-teal flex items-center gap-1">
                              <Sparkles className="w-3.5 h-3.5 text-teal" />
                              {rec.label}
                            </span>
                            {!isRecSelected && (
                              <button
                                type="button"
                                onClick={() => {
                                  setMissingConfigs((prev) => ({
                                    ...prev,
                                    [col]: {
                                      strategy: rec.strategy,
                                      value: rec.defaultConstant || '',
                                    },
                                  }));
                                }}
                                className="text-[10px] text-navy/60 hover:text-teal underline font-mono"
                              >
                                Re-apply
                              </button>
                            )}
                          </div>
                          <p className="text-[11px] text-navy/70 leading-relaxed">{rec.reason}</p>
                          {rec.stats && (
                            <span className="text-[10px] text-navy/50 font-mono block">
                              Evidence: {rec.stats}
                            </span>
                          )}
                        </div>
                      </td>

                      {/* Strategy Selection Dropdown */}
                      <td className="p-2.5 font-sans">
                        <div className="space-y-1">
                          <select
                            value={currentConf.strategy || 'none'}
                            onChange={(e) => {
                              const val = e.target.value;
                              setMissingConfigs((prev) => ({
                                ...prev,
                                [col]: {
                                  ...prev[col],
                                  strategy: val,
                                },
                              }));
                            }}
                            className="px-2 py-1 bg-white border border-stone rounded text-xs text-navy focus:outline-none focus:border-teal"
                          >
                            <option value="none">Leave Unchanged</option>
                            {!isEmpty && isNumeric && !isId && (
                              <option value="mean">Mean (Average)</option>
                            )}
                            {!isEmpty && isNumeric && !isId && (
                              <option value="median">Median (Middle Value)</option>
                            )}
                            {!isEmpty && !isId && (
                              <option value="mode">Mode (Most Frequent)</option>
                            )}
                            {!isEmpty && <option value="constant">Custom Constant Value</option>}
                          </select>

                          <div>
                            {isRecSelected && !isNone && (
                              <Badge variant="teal" size="sm">
                                Recommended
                              </Badge>
                            )}
                            {!isRecSelected && !isNone && (
                              <Badge variant="amber" size="sm">
                                Customized
                              </Badge>
                            )}
                            {isNone && isAffected && (
                              <Badge variant="stone" size="sm">
                                Leave Unchanged
                              </Badge>
                            )}
                          </div>
                        </div>
                      </td>

                      {/* Custom Constant Input */}
                      <td className="p-2.5 font-sans">
                        {currentConf.strategy === 'constant' ? (
                          <input
                            type={isNumeric ? 'number' : 'text'}
                            placeholder={isNumeric ? 'e.g. 0' : 'e.g. Unknown'}
                            value={currentConf.value || ''}
                            onChange={(e) => {
                              const val = e.target.value;
                              setMissingConfigs((prev) => ({
                                ...prev,
                                [col]: {
                                  ...prev[col],
                                  value: isNumeric && val !== '' ? Number(val) : val,
                                },
                              }));
                            }}
                            className="px-2 py-1 bg-white border border-stone rounded text-xs text-navy w-32 focus:outline-none focus:border-teal"
                          />
                        ) : (
                          <span className="text-navy/40 text-[11px]">—</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </Card>

      {/* SECTION 2: Invalid Numerical Value Handling */}
      <Card
        title="2. Invalid Numerical Values & Domain Boundaries"
        subtitle="Safe handling for infinite values (+inf / -inf) is preselected. Minimum and maximum domain limits remain optional."
      >
        <div className="space-y-4 text-xs font-sans">
          <p className="text-navy/70">
            Detected infinities are preselected to be converted into missing values (NaN), allowing them to be imputed safely alongside other missing data. You can adjust the action or define domain min/max boundaries.
          </p>

          <div className="space-y-3">
            {columnsList
              .filter((col) => isNumericColumn(col))
              .map((col) => {
                const conf = invalidNumConfigs[col] || {
                  action: 'none',
                  min_value: '',
                  max_value: '',
                  handle_infinities: true,
                };
                const infInfo = columnsWithInfinities.find((c) => c.column === col);
                const isRangeInvalid =
                  conf.min_value !== '' &&
                  conf.max_value !== '' &&
                  Number(conf.min_value) > Number(conf.max_value);

                return (
                  <div
                    key={col}
                    className="p-3 bg-stone-light/30 border border-stone rounded-sm space-y-2.5 font-sans"
                  >
                    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <span className="font-semibold text-navy font-mono">{col}</span>
                        <Badge variant="teal" size="sm">
                          {columnTypes[col] || 'numeric'}
                        </Badge>
                        {infInfo ? (
                          <Badge variant="red" size="sm">
                            {infInfo.total} Infinite (+inf / -inf) Detected
                          </Badge>
                        ) : (
                          <Badge variant="stone" size="sm">
                            No Infinities
                          </Badge>
                        )}
                      </div>

                      {/* Action selector */}
                      <div className="flex items-center gap-2">
                        <span className="text-[11px] text-navy/60">Action:</span>
                        <select
                          value={conf.action || 'none'}
                          onChange={(e) => {
                            const act = e.target.value;
                            setInvalidNumConfigs((prev) => ({
                              ...prev,
                              [col]: { ...prev[col], action: act },
                            }));
                          }}
                          className="px-2 py-1 bg-white border border-stone rounded text-xs text-navy focus:outline-none focus:border-teal"
                        >
                          <option value="none">Leave Unchanged</option>
                          <option value="replace_with_nan">Replace with Missing (NaN) — Recommended</option>
                          <option value="drop_rows">Remove Entire Rows (⚠️ Deletes Data)</option>
                        </select>
                      </div>
                    </div>

                    {/* Range settings when action is active */}
                    {conf.action !== 'none' && (
                      <div className="pt-2 border-t border-stone/60 grid grid-cols-1 sm:grid-cols-3 gap-3 font-mono text-xs">
                        <div>
                          <label className="text-[10px] text-navy/60 uppercase block mb-1">
                            Min Acceptable Value (Optional)
                          </label>
                          <input
                            type="number"
                            placeholder="e.g. 0"
                            value={conf.min_value ?? ''}
                            onChange={(e) => {
                              const val = e.target.value;
                              setInvalidNumConfigs((prev) => ({
                                ...prev,
                                [col]: { ...prev[col], min_value: val },
                              }));
                            }}
                            className="w-full px-2 py-1 bg-white border border-stone rounded text-xs text-navy focus:outline-none focus:border-teal"
                          />
                        </div>

                        <div>
                          <label className="text-[10px] text-navy/60 uppercase block mb-1">
                            Max Acceptable Value (Optional)
                          </label>
                          <input
                            type="number"
                            placeholder="e.g. 100"
                            value={conf.max_value ?? ''}
                            onChange={(e) => {
                              const val = e.target.value;
                              setInvalidNumConfigs((prev) => ({
                                ...prev,
                                [col]: { ...prev[col], max_value: val },
                              }));
                            }}
                            className="w-full px-2 py-1 bg-white border border-stone rounded text-xs text-navy focus:outline-none focus:border-teal"
                          />
                        </div>

                        <div className="flex items-center pt-4 font-sans">
                          <label className="flex items-center gap-2 cursor-pointer text-navy text-xs">
                            <input
                              type="checkbox"
                              checked={conf.handle_infinities ?? true}
                              onChange={(e) => {
                                setInvalidNumConfigs((prev) => ({
                                  ...prev,
                                  [col]: { ...prev[col], handle_infinities: e.target.checked },
                                }));
                              }}
                              className="rounded border-stone text-teal focus:ring-teal"
                            />
                            <span>Include infinities (+inf / -inf)</span>
                          </label>
                        </div>

                        {isRangeInvalid && (
                          <div className="sm:col-span-3 text-red-600 text-xs font-sans">
                            ⚠️ Error: Minimum value cannot be greater than maximum value.
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
          </div>
        </div>
      </Card>

      {/* SECTION 3: Categorical Value Mapping */}
      <Card
        title="3. Categorical Value Mapping & Normalization"
        subtitle="Detected casing and whitespace discrepancies are preselected for your review. Spelling modifications require explicit confirmation."
      >
        <div className="space-y-4 text-xs font-sans">
          <p className="text-navy/70">
            Detected variations (e.g. 'india', 'INDIA' → 'India') are pre-populated with their canonical target. Review each mapping below; uncheck any pair you do not wish to normalize.
          </p>

          {/* Grouped by column */}
          {columnsList
            .filter((col) => isTextColumn(col))
            .map((col) => {
              const mappingsList = valueMappings[col] || [];

              return (
                <div key={col} className="p-3 bg-stone-light/30 border border-stone rounded-sm space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-navy font-mono">{col}</span>
                      <Badge variant="stone" size="sm">
                        {columnTypes[col] || 'text'}
                      </Badge>
                      {mappingsList.length > 0 && (
                        <span className="text-[11px] text-navy/60 font-mono">
                          {mappingsList.filter((m) => m.approved).length} / {mappingsList.length} selected for review
                        </span>
                      )}
                    </div>

                    <button
                      type="button"
                      onClick={() => {
                        setValueMappings((prev) => ({
                          ...prev,
                          [col]: [...(prev[col] || []), { from: '', to: '', approved: true, isCustom: true }],
                        }));
                      }}
                      className="text-xs text-teal hover:underline flex items-center gap-1 font-semibold"
                    >
                      <Plus className="w-3.5 h-3.5" />
                      Add Custom Mapping
                    </button>
                  </div>

                  {mappingsList.length > 0 ? (
                    <div className="space-y-2">
                      {mappingsList.map((item, idx) => (
                        <div
                          key={idx}
                          className="flex flex-col sm:flex-row sm:items-center gap-2 p-2 bg-white border border-stone/70 rounded-sm font-mono text-xs"
                        >
                          <label className="flex items-center gap-2 cursor-pointer shrink-0">
                            <input
                              type="checkbox"
                              checked={item.approved}
                              onChange={(e) => {
                                const chk = e.target.checked;
                                setValueMappings((prev) => {
                                  const updated = [...(prev[col] || [])];
                                  updated[idx] = { ...updated[idx], approved: chk };
                                  return { ...prev, [col]: updated };
                                });
                              }}
                              className="rounded border-stone text-teal focus:ring-teal"
                            />
                            <span className="text-[11px] text-navy/50 font-sans">Selected for review:</span>
                          </label>

                          <div className="flex items-center gap-2 flex-1">
                            <input
                              type="text"
                              placeholder="Original value"
                              value={item.from}
                              onChange={(e) => {
                                const val = e.target.value;
                                setValueMappings((prev) => {
                                  const updated = [...(prev[col] || [])];
                                  updated[idx] = { ...updated[idx], from: val };
                                  return { ...prev, [col]: updated };
                                });
                              }}
                              className="flex-1 px-2 py-1 bg-stone-light/20 border border-stone rounded text-xs text-navy focus:outline-none focus:border-teal"
                            />

                            <ArrowRight className="w-4 h-4 text-navy/40 shrink-0" />

                            <input
                              type="text"
                              placeholder="Canonical value"
                              value={item.to}
                              onChange={(e) => {
                                const val = e.target.value;
                                setValueMappings((prev) => {
                                  const updated = [...(prev[col] || [])];
                                  updated[idx] = { ...updated[idx], to: val };
                                  return { ...prev, [col]: updated };
                                });
                              }}
                              className="flex-1 px-2 py-1 bg-stone-light/20 border border-stone rounded text-xs text-navy focus:outline-none focus:border-teal font-semibold text-teal-dark"
                            />
                          </div>

                          <button
                            type="button"
                            onClick={() => {
                              setValueMappings((prev) => ({
                                ...prev,
                                [col]: prev[col].filter((_, i) => i !== idx),
                              }));
                            }}
                            className="text-navy/40 hover:text-red-600 p-1"
                            title="Remove mapping"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-[11px] text-navy/50 font-sans italic">
                      No inconsistent category variants flagged for this column. You may add custom mappings above if needed.
                    </p>
                  )}
                </div>
              );
            })}
        </div>
      </Card>

      {/* SECTION 4: Duplicate Row Removal */}
      <Card
        title="4. Duplicate Row Removal"
        subtitle="Optional recommendation: select this operation to remove redundant identical records."
      >
        <div className="space-y-4 text-xs font-sans">
          <label className="flex items-start gap-3 cursor-pointer">
            <input
              type="checkbox"
              checked={removeDuplicates}
              onChange={(e) => setRemoveDuplicates(e.target.checked)}
              className="mt-0.5 rounded border-stone text-teal focus:ring-teal"
            />
            <div>
              <div className="flex items-center gap-2">
                <span className="font-semibold text-navy block">Remove Duplicate Rows</span>
                {qualityData?.duplicate_rows?.duplicate_count > 0 && (
                  <Badge variant="amber" size="sm">
                    {qualityData.duplicate_rows.duplicate_count} Duplicate(s) Found
                  </Badge>
                )}
              </div>
              <span className="text-navy/60 text-[11px] block mt-0.5">
                This is not selected by default. When selected, the first occurrence of each duplicate record is preserved, and subsequent duplicates are removed.
              </span>
            </div>
          </label>

          {removeDuplicates && (
            <div className="ml-6 pl-4 border-l-2 border-stone space-y-3 pt-1">
              <span className="font-semibold text-navy text-[11px] block uppercase font-mono">
                DUPLICATE MATCHING SCOPE:
              </span>
              <div className="space-y-2">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="radio"
                    name="dup_scope"
                    checked={!useDuplicateSubset}
                    onChange={() => {
                      setUseDuplicateSubset(false);
                      setDuplicateSubset([]);
                    }}
                    className="text-teal focus:ring-teal"
                  />
                  <span className="text-navy">Match across all columns (entire row must be identical)</span>
                </label>

                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="radio"
                    name="dup_scope"
                    checked={useDuplicateSubset}
                    onChange={() => setUseDuplicateSubset(true)}
                    className="text-teal focus:ring-teal"
                  />
                  <span className="text-navy">Match by specific key columns only</span>
                </label>
              </div>

              {useDuplicateSubset && (
                <div className="pt-2 space-y-2">
                  <span className="text-[11px] text-navy/60 block">Select key columns to identify duplicates:</span>
                  <div className="flex flex-wrap gap-2">
                    {columnsList.map((col) => {
                      const isSelected = duplicateSubset.includes(col);
                      return (
                        <button
                          key={col}
                          type="button"
                          onClick={() => {
                            setDuplicateSubset((prev) =>
                              isSelected ? prev.filter((c) => c !== col) : [...prev, col]
                            );
                          }}
                          className={`px-2.5 py-1 text-xs font-mono rounded border transition-colors ${
                            isSelected
                              ? 'bg-teal text-white border-teal'
                              : 'bg-white text-navy/70 border-stone hover:border-navy/30'
                          }`}
                        >
                          {col}
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </Card>

      {/* SECTION 5: Text Formatting & Normalization */}
      <Card
        title="5. Text Formatting & Normalization"
        subtitle="Clean up trailing spaces and normalize letter casing on categorical columns."
      >
        <div className="space-y-4 text-xs font-sans">
          {/* Whitespace trimming */}
          <div className="space-y-2">
            <label className="flex items-start gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={trimAllWhitespace}
                onChange={(e) => {
                  setTrimAllWhitespace(e.target.checked);
                  if (e.target.checked) setSelectedTrimColumns([]);
                }}
                className="mt-0.5 rounded border-stone text-teal focus:ring-teal"
              />
              <div>
                <span className="font-semibold text-navy block">Trim Whitespace on All Text Columns</span>
                <span className="text-navy/60 text-[11px] block mt-0.5">
                  Removes accidental leading and trailing spaces (e.g., ' New York ' becomes 'New York').
                </span>
              </div>
            </label>
          </div>

          {/* Casing Normalization */}
          <div className="pt-3 border-t border-stone space-y-3">
            <span className="font-semibold text-navy block">Consistent Letter Casing</span>
            <p className="text-navy/60 text-[11px]">
              Convert text values to uniform lowercase or uppercase to prevent accidental category splits.
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3 font-mono">
              {columnsList
                .filter((col) => isTextColumn(col))
                .map((col) => {
                  const currentCase = casingConfigs[col] || 'none';
                  return (
                    <div key={col} className="p-2.5 bg-stone-light/30 border border-stone rounded-sm">
                      <span className="font-semibold text-navy block mb-1.5 truncate" title={col}>
                        {col}
                      </span>
                      <select
                        value={currentCase}
                        onChange={(e) => {
                          const val = e.target.value;
                          setCasingConfigs((prev) => ({
                            ...prev,
                            [col]: val,
                          }));
                        }}
                        className="w-full px-2 py-1 bg-white border border-stone rounded text-xs text-navy font-sans focus:outline-none focus:border-teal"
                      >
                        <option value="none">Keep Original Casing</option>
                        <option value="lower">lowercase (all small)</option>
                        <option value="upper">UPPERCASE (all caps)</option>
                      </select>
                    </div>
                  );
                })}
            </div>
          </div>
        </div>
      </Card>

      {/* SECTION 6: Column Removal */}
      <Card
        title="6. Column Pruning"
        subtitle="Optional recommendations: select either operation to remove columns from the cleaned dataset."
      >
        <div className="space-y-3 text-xs font-sans">
          <label className="flex items-start gap-3 cursor-pointer">
            <input
              type="checkbox"
              checked={dropEmptyColumns}
              onChange={(e) => setDropEmptyColumns(e.target.checked)}
              className="mt-0.5 rounded border-stone text-teal focus:ring-teal"
            />
            <div>
              <div className="flex items-center gap-2">
                <span className="font-semibold text-navy block">Drop Completely Empty Columns</span>
                {qualityData?.empty_columns?.length > 0 && (
                  <Badge variant="red" size="sm">
                    {qualityData.empty_columns.length} Empty Column(s)
                  </Badge>
                )}
              </div>
              <span className="text-navy/60 text-[11px] block mt-0.5">
                Optional and unchecked by default. Select to remove columns where 100% of the rows are missing or null.
              </span>
            </div>
          </label>

          <label className="flex items-start gap-3 cursor-pointer">
            <input
              type="checkbox"
              checked={dropConstantColumns}
              onChange={(e) => setDropConstantColumns(e.target.checked)}
              className="mt-0.5 rounded border-stone text-teal focus:ring-teal"
            />
            <div>
              <div className="flex items-center gap-2">
                <span className="font-semibold text-navy block">Drop Constant (Zero-Variance) Columns</span>
                {qualityData?.constant_columns?.length > 0 && (
                  <Badge variant="amber" size="sm">
                    {qualityData.constant_columns.length} Constant Column(s)
                  </Badge>
                )}
              </div>
              <span className="text-navy/60 text-[11px] block mt-0.5">
                Optional and unchecked by default. Select to remove columns where every single row contains the exact same non-null value.
              </span>
            </div>
          </label>
        </div>
      </Card>

      {/* SECTION 7: Review & Confirm Cleaning Operations */}
      <Card
        title="7. Review & Confirm Cleaning Operations"
        subtitle="Verify your selections before applying transformations. The original dataset remains untouched."
      >
        <div className="space-y-4 text-xs font-sans">
          {hasSelectedOperations ? (
            <div className="p-3.5 bg-stone-light/40 border border-stone rounded-sm space-y-2">
              <span className="font-semibold text-navy block">
                Selected Cleaning Plan ({operationsCountSummary} operation(s)):
              </span>
              <ul className="list-disc list-inside space-y-1 font-mono text-navy/80 text-[11px]">
                {compiledOperations.missing_values &&
                  Object.entries(compiledOperations.missing_values).map(([col, strat]) => (
                    <li key={col}>
                      Impute missing values in '{col}' using{' '}
                      <span className="font-bold text-teal">
                        {typeof strat === 'object' ? `constant (${strat.value})` : strat}
                      </span>
                    </li>
                  ))}
                {compiledOperations.invalid_numerical_values &&
                  Object.entries(compiledOperations.invalid_numerical_values).map(([col, cfg]) => (
                    <li key={col}>
                      Invalid numbers in '{col}':{' '}
                      <span className="font-bold text-teal">{cfg.action}</span>
                      {cfg.min_value !== undefined && ` [min: ${cfg.min_value}]`}
                      {cfg.max_value !== undefined && ` [max: ${cfg.max_value}]`}
                    </li>
                  ))}
                {compiledOperations.value_mapping &&
                  Object.entries(compiledOperations.value_mapping).map(([col, map]) => (
                    <li key={col}>
                      Map {Object.keys(map).length} value(s) in '{col}':{' '}
                      <span className="font-bold text-teal">
                        {Object.entries(map)
                          .map(([f, t]) => `"${f}"→"${t}"`)
                          .join(', ')}
                      </span>
                    </li>
                  ))}
                {compiledOperations.remove_duplicates && (
                  <li>
                    Remove duplicate rows{' '}
                    <span className="font-bold text-teal">
                      (subset: {compiledOperations.duplicate_subset ? compiledOperations.duplicate_subset.join(', ') : 'all columns'})
                    </span>
                  </li>
                )}
                {compiledOperations.trim_whitespace && (
                  <li>
                    Trim leading/trailing whitespace{' '}
                    <span className="font-bold text-teal">
                      ({Array.isArray(compiledOperations.trim_whitespace) ? compiledOperations.trim_whitespace.join(', ') : 'all text columns'})
                    </span>
                  </li>
                )}
                {compiledOperations.casing &&
                  Object.entries(compiledOperations.casing).map(([col, c]) => (
                    <li key={col}>
                      Convert column '{col}' to <span className="font-bold text-teal">{c}case</span>
                    </li>
                  ))}
                {compiledOperations.drop_empty_columns && (
                  <li>
                    Drop <span className="font-bold text-teal">completely empty columns</span>
                  </li>
                )}
                {compiledOperations.drop_constant_columns && (
                  <li>
                    Drop <span className="font-bold text-teal">constant zero-variance columns</span>
                  </li>
                )}
              </ul>
            </div>
          ) : (
            <div className="p-4 bg-stone-light/30 border border-stone rounded-sm text-navy/60 text-center">
              No cleaning operations selected. Click "Apply All Recommendations" above or make manual selections.
            </div>
          )}

          {willDeleteRows && (
            <div className="p-3 bg-amber-50 border border-amber-200 text-amber-800 rounded-sm text-xs flex items-start gap-2">
              <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
              <div>
                <span className="font-semibold">Row Deletion Warning:</span>
                <span className="ml-1">
                  Your selected plan includes operations that remove rows (duplicate removal and/or dropping invalid numerical rows). Ensure you review these criteria before confirming.
                </span>
              </div>
            </div>
          )}

          {cleaningError && (
            <div className="p-3 bg-red-50 border border-red-200 text-red-800 rounded-sm text-xs flex items-start gap-2">
              <AlertTriangle className="w-4 h-4 text-red-600 shrink-0 mt-0.5" />
              <div>
                <span className="font-semibold">Cleaning Failed:</span>
                <span className="ml-1">{cleaningError}</span>
              </div>
            </div>
          )}

          <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-2">
            <div className="flex items-center gap-2 text-navy/60 text-[11px]">
              <ShieldCheck className="w-4 h-4 text-teal shrink-0" />
              <span>Original dataset will remain 100% intact and unchanged.</span>
            </div>

            <Button
              variant="primary"
              size="md"
              icon={CheckCircle2}
              disabled={!hasSelectedOperations || isCleaning}
              loading={isCleaning}
              onClick={() => setShowApprovalModal(true)}
            >
              Review & Approve Cleaning Plan
            </Button>
          </div>
        </div>
      </Card>

      {/* Confirmation Modal */}
      {showApprovalModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-navy/60 backdrop-blur-xs">
          <div className="bg-white border border-stone rounded-sm shadow-xl max-w-lg w-full p-6 space-y-4">
            <div className="flex items-start gap-3">
              <div className="p-2 bg-sage/40 rounded border border-teal/30">
                <ShieldCheck className="w-6 h-6 text-teal" />
              </div>
              <div>
                <h3 className="font-serif text-lg font-bold text-navy">Approve Dataset Cleaning</h3>
                <p className="text-xs text-navy/70 font-sans mt-0.5">
                  Please review the operations to be executed on your dataset.
                </p>
              </div>
            </div>

            <div className="p-3 bg-stone-light/40 border border-stone rounded text-xs space-y-2 font-mono max-h-60 overflow-y-auto">
              <span className="font-sans font-semibold text-navy text-[11px] uppercase block">
                OPERATIONS TO BE PERFORMED:
              </span>
              <ul className="list-disc list-inside space-y-1 text-navy/80 text-[11px]">
                {compiledOperations.missing_values && (
                  <li>Impute {Object.keys(compiledOperations.missing_values).length} column(s)</li>
                )}
                {compiledOperations.invalid_numerical_values && (
                  <li>Handle invalid numerical values in {Object.keys(compiledOperations.invalid_numerical_values).length} column(s)</li>
                )}
                {compiledOperations.value_mapping && (
                  <li>Apply categorical mappings in {Object.keys(compiledOperations.value_mapping).length} column(s)</li>
                )}
                {compiledOperations.remove_duplicates && <li>Remove duplicate records</li>}
                {compiledOperations.trim_whitespace && <li>Trim leading and trailing whitespace</li>}
                {compiledOperations.casing && (
                  <li>Normalize casing on {Object.keys(compiledOperations.casing).length} column(s)</li>
                )}
                {compiledOperations.drop_empty_columns && <li>Drop empty columns</li>}
                {compiledOperations.drop_constant_columns && <li>Drop constant columns</li>}
              </ul>
            </div>

            {willDeleteRows && (
              <p className="text-amber-800 text-[11px] font-semibold bg-amber-50 p-2 rounded border border-amber-200">
                ⚠️ Note: This cleaning plan will remove matching rows from the cleaned dataset.
              </p>
            )}

            <p className="text-[11px] text-navy/60 font-sans">
              <strong>Notice:</strong> The backend executes transformations directly upon confirmation. Your original file remains untouched, and a separate cleaned dataset will be created and saved.
            </p>

            <div className="flex items-center justify-end gap-3 pt-3 border-t border-stone">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowApprovalModal(false)}
                disabled={isCleaning}
              >
                Go Back & Edit
              </Button>
              <Button
                variant="primary"
                size="sm"
                icon={CheckCircle2}
                onClick={handleExecuteClean}
                loading={isCleaning}
              >
                Confirm & Clean Dataset
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
