import React, { useState, useRef, useCallback } from 'react';
import { UploadCloud, FileSpreadsheet, AlertCircle, CheckCircle2, X, RefreshCw } from 'lucide-react';
import Button from '../common/Button';
import Badge from '../common/Badge';
import { uploadDataset } from '../../api/datasetApi';
import { getErrorMessage } from '../../api/client';
import { formatBytes } from '../../utils/formatters';
import { useDataset } from '../../context/DatasetContext';

const MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024; // 50 MB

export default function CsvUploader() {
  const { setActiveDataset, activeDataset } = useDataset();

  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [validationError, setValidationError] = useState(null);
  const [serverError, setServerError] = useState(null);
  const [uploadSuccess, setUploadSuccess] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  const fileInputRef = useRef(null);
  const abortControllerRef = useRef(null);

  /**
   * Client-side validation for selected files before dispatch.
   */
  const validateFile = (file) => {
    if (!file) {
      return 'No file selected. Please choose a CSV file.';
    }

    if (!file.name || !file.name.toLowerCase().endsWith('.csv')) {
      return 'Please choose a CSV file (.csv). Other file formats are not supported.';
    }

    if (file.size === 0) {
      return 'The selected file is empty (0 bytes). Please choose a valid CSV dataset.';
    }

    if (file.size > MAX_FILE_SIZE_BYTES) {
      return `File size (${formatBytes(file.size)}) exceeds the maximum allowed limit of 50 MB.`;
    }

    return null;
  };

  const handleFileSelection = (file) => {
    setValidationError(null);
    setServerError(null);
    setUploadSuccess(null);

    const error = validateFile(file);
    if (error) {
      setValidationError(error);
      setSelectedFile(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
      return;
    }

    setSelectedFile(file);
  };

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      e.stopPropagation();
      setDragActive(false);

      if (isUploading) return;

      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFileSelection(e.dataTransfer.files[0]);
      }
    },
    [isUploading]
  );

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFileSelection(e.target.files[0]);
    }
  };

  const resetSelection = () => {
    if (isUploading && abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setSelectedFile(null);
    setValidationError(null);
    setServerError(null);
    setIsUploading(false);
    setUploadProgress(0);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleUpload = async () => {
    if (!selectedFile || isUploading) return;

    // Double-check validation
    const error = validateFile(selectedFile);
    if (error) {
      setValidationError(error);
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);
    setServerError(null);
    setUploadSuccess(null);

    abortControllerRef.current = new AbortController();

    try {
      const response = await uploadDataset(
        selectedFile,
        (progress) => setUploadProgress(progress),
        abortControllerRef.current.signal
      );

      // Successfully processed by backend
      const data = response.data;
      setUploadSuccess({
        filename: data.filename || selectedFile.name,
        dataset_id: data.dataset_id,
        row_count: data.row_count,
        column_count: data.column_count,
        columns: data.columns || [],
      });

      // Save into DatasetContext & sessionStorage
      setActiveDataset({
        dataset_id: data.dataset_id,
        filename: data.filename || selectedFile.name,
        row_count: data.row_count,
        column_count: data.column_count,
        columns: data.columns || [],
        uploaded_at: Date.now(),
      });

      setSelectedFile(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
    } catch (err) {
      if (err.name === 'CanceledError' || err.message === 'canceled') {
        setServerError('Upload was cancelled.');
      } else {
        const errorMsg = getErrorMessage(err, 'Failed to upload CSV file. Please check the file and try again.');
        setServerError(errorMsg);
      }
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  return (
    <div className="space-y-4">
      {/* Hidden native input */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".csv,text/csv,application/vnd.ms-excel"
        onChange={handleChange}
        className="hidden"
        disabled={isUploading}
      />

      {/* Dropzone container */}
      <div
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        className={`border-2 border-dashed rounded-sm transition-all duration-150 p-8 text-center select-none ${
          dragActive
            ? 'border-teal bg-teal-soft/40 ring-2 ring-teal/20'
            : selectedFile
            ? 'border-stone-dark bg-stone-light/30'
            : 'border-stone hover:border-teal/60 bg-white hover:bg-stone-light/20'
        }`}
      >
        {!selectedFile ? (
          <div className="space-y-3">
            <div className="w-12 h-12 mx-auto rounded-full bg-sage/60 flex items-center justify-center text-teal">
              <UploadCloud className="w-6 h-6" />
            </div>

            <div>
              <p className="font-serif text-lg font-semibold text-navy">
                Drag and drop your CSV file here
              </p>
              <p className="text-xs text-navy/60 font-sans mt-0.5">
                or click below to choose a file from your computer
              </p>
            </div>

            <div className="pt-2">
              <Button
                variant="outline"
                size="md"
                onClick={() => fileInputRef.current?.click()}
                disabled={isUploading}
              >
                Browse Files
              </Button>
            </div>

            <p className="text-[11px] text-navy/50 font-mono">
              Accepted format: CSV (.csv) • Maximum size: 50 MB
            </p>
          </div>
        ) : (
          /* Selected File Preview & Controls */
          <div className="space-y-4 max-w-lg mx-auto text-left">
            <div className="flex items-start justify-between gap-3 p-3.5 bg-white border border-stone rounded-sm">
              <div className="flex items-center gap-3 min-w-0">
                <div className="p-2 rounded bg-sage/70 text-teal shrink-0">
                  <FileSpreadsheet className="w-5 h-5" />
                </div>
                <div className="min-w-0">
                  <p className="text-sm font-semibold text-navy truncate" title={selectedFile.name}>
                    {selectedFile.name}
                  </p>
                  <p className="text-xs text-navy/60 font-mono">
                    {formatBytes(selectedFile.size)}
                  </p>
                </div>
              </div>

              {!isUploading && (
                <button
                  type="button"
                  onClick={resetSelection}
                  className="p-1 hover:bg-stone-light rounded text-navy/50 hover:text-navy transition-colors shrink-0"
                  title="Remove selection"
                >
                  <X className="w-4 h-4" />
                </button>
              )}
            </div>

            {/* Upload Progress Bar */}
            {isUploading && (
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs font-mono text-navy/70">
                  <span>Uploading to server...</span>
                  <span>{uploadProgress}%</span>
                </div>
                <div className="w-full h-2 bg-stone rounded-full overflow-hidden">
                  <div
                    className="h-full bg-teal transition-all duration-150 rounded-full"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
              </div>
            )}

            {/* Actions */}
            <div className="flex items-center justify-end gap-3 pt-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={resetSelection}
                disabled={isUploading && uploadProgress > 95}
              >
                {isUploading ? 'Cancel' : 'Choose Different File'}
              </Button>
              <Button
                variant="primary"
                size="md"
                onClick={handleUpload}
                loading={isUploading}
                disabled={isUploading}
              >
                {isUploading ? `Uploading (${uploadProgress}%)` : 'Upload & Verify CSV'}
              </Button>
            </div>
          </div>
        )}
      </div>

      {/* Validation Error Banner */}
      {validationError && (
        <div className="p-3.5 bg-red-50 border border-red-200 rounded-sm text-xs text-red-800 flex items-start gap-2.5">
          <AlertCircle className="w-4 h-4 text-red-600 shrink-0 mt-0.5" />
          <div className="space-y-0.5">
            <p className="font-semibold">Unable to accept file</p>
            <p className="text-red-700/90">{validationError}</p>
          </div>
        </div>
      )}

      {/* Server Error Banner */}
      {serverError && (
        <div className="p-3.5 bg-red-50 border border-red-200 rounded-sm text-xs text-red-800 flex items-start gap-2.5">
          <AlertCircle className="w-4 h-4 text-red-600 shrink-0 mt-0.5" />
          <div className="space-y-0.5">
            <p className="font-semibold">Upload failed</p>
            <p className="text-red-700/90">{serverError}</p>
          </div>
        </div>
      )}

      {/* Upload Success Banner */}
      {uploadSuccess && (
        <div className="p-4 bg-sage/50 border border-sage-dark/60 rounded-sm text-xs text-navy space-y-2">
          <div className="flex items-center gap-2 text-teal-dark font-semibold">
            <CheckCircle2 className="w-4 h-4 shrink-0 text-teal" />
            <span>CSV file uploaded and verified successfully!</span>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1 font-mono text-[11px]">
            <div className="bg-white/80 p-2 rounded border border-stone">
              <span className="text-navy/50 font-sans block text-[10px]">FILE</span>
              <span className="font-semibold truncate block" title={uploadSuccess.filename}>
                {uploadSuccess.filename}
              </span>
            </div>
            <div className="bg-white/80 p-2 rounded border border-stone">
              <span className="text-navy/50 font-sans block text-[10px]">ROWS</span>
              <span className="font-semibold">
                {uploadSuccess.row_count ? uploadSuccess.row_count.toLocaleString() : '—'}
              </span>
            </div>
            <div className="bg-white/80 p-2 rounded border border-stone">
              <span className="text-navy/50 font-sans block text-[10px]">COLUMNS</span>
              <span className="font-semibold">{uploadSuccess.column_count ?? '—'}</span>
            </div>
            <div className="bg-white/80 p-2 rounded border border-stone">
              <span className="text-navy/50 font-sans block text-[10px]">SESSION ID</span>
              <span className="font-semibold truncate block" title={uploadSuccess.dataset_id}>
                {uploadSuccess.dataset_id}
              </span>
            </div>
          </div>
          {Array.isArray(uploadSuccess.columns) && uploadSuccess.columns.length > 0 && (
            <div className="pt-1">
              <span className="text-[11px] text-navy/60 block mb-1">Detected Columns:</span>
              <div className="flex flex-wrap gap-1.5 max-h-24 overflow-y-auto p-1.5 bg-white/60 rounded border border-stone/60">
                {uploadSuccess.columns.map((col) => (
                  <Badge key={col} variant="stone" size="sm">
                    {col}
                  </Badge>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
