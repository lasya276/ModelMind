import React from 'react';
import {
  CartesianGrid,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

const COLORS = ['#137C78', '#D18B45', '#54708F', '#986C8B', '#6D8B64', '#B55D54'];

function RegressionChart({ points, bounds }) {
  return (
    <div className="h-72 w-full" role="img" aria-label="Scatter plot comparing actual and predicted test values">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 12, right: 20, bottom: 20, left: 12 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#E4E2D9" />
          <XAxis type="number" dataKey="actual" name="Actual" domain={bounds} label={{ value: 'Actual', position: 'insideBottom', offset: -12 }} tick={{ fontSize: 10 }} />
          <YAxis type="number" dataKey="predicted" name="Predicted" domain={bounds} label={{ value: 'Predicted', angle: -90, position: 'insideLeft' }} tick={{ fontSize: 10 }} />
          <Tooltip
            cursor={{ strokeDasharray: '3 3' }}
            formatter={(value) => Number(value).toPrecision(5)}
            labelFormatter={(_, payload) => payload?.[0]?.payload?.rowIndex !== undefined
              ? `Dataset row ${payload[0].payload.rowIndex}`
              : ''}
          />
          <ReferenceLine segment={[{ x: bounds[0], y: bounds[0] }, { x: bounds[1], y: bounds[1] }]} stroke="#9AA6A8" strokeDasharray="5 5" />
          <Scatter name="Test observations" data={points} fill="#137C78" />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

function ClusteringChart({ rows, featureColumns, clusterIds, xFeature, yFeature }) {
  const plotRows = rows.map((row) => ({
    ...row.features,
    rowIndex: row.row_index,
    cluster: row.cluster,
    plotY: featureColumns.length > 1 ? row.features[yFeature] : row.cluster,
  }));
  return (
    <div className="h-72 w-full" role="img" aria-label="Scatter plot of observations grouped by K-Means cluster">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 12, right: 20, bottom: 20, left: 12 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#E4E2D9" />
          <XAxis type="number" dataKey={xFeature} name={xFeature} label={{ value: xFeature, position: 'insideBottom', offset: -12 }} tick={{ fontSize: 10 }} />
          <YAxis type="number" dataKey={featureColumns.length > 1 ? yFeature : 'plotY'} name={featureColumns.length > 1 ? yFeature : 'Cluster'} label={{ value: featureColumns.length > 1 ? yFeature : 'Cluster', angle: -90, position: 'insideLeft' }} tick={{ fontSize: 10 }} />
          <Tooltip
            formatter={(value) => Number(value).toPrecision(5)}
            labelFormatter={(_, payload) => payload?.[0]?.payload?.rowIndex !== undefined
              ? `Dataset row ${payload[0].payload.rowIndex} · Cluster ${payload[0].payload.cluster}`
              : ''}
          />
          <Legend />
          {clusterIds.map((cluster, index) => (
            <Scatter
              key={cluster}
              name={`Cluster ${cluster}`}
              data={plotRows.filter((row) => String(row.cluster) === cluster)}
              fill={COLORS[index % COLORS.length]}
            />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

function AnomalyChart({ normalRows, anomalyRows }) {
  return (
    <div className="h-72 w-full" role="img" aria-label="Anomaly score scatter plot by dataset row">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 12, right: 20, bottom: 20, left: 12 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#E4E2D9" />
          <XAxis type="number" dataKey="row_index" name="Dataset row" label={{ value: 'Dataset row', position: 'insideBottom', offset: -12 }} tick={{ fontSize: 10 }} />
          <YAxis type="number" dataKey="score" name="Decision-function score" label={{ value: 'Score', angle: -90, position: 'insideLeft' }} tick={{ fontSize: 10 }} />
          <ReferenceLine y={0} stroke="#9AA6A8" strokeDasharray="5 5" />
          <Tooltip
            formatter={(value) => Number(value).toPrecision(5)}
            labelFormatter={(_, payload) => payload?.[0]?.payload?.is_anomaly
              ? `Dataset row ${payload[0].payload.row_index} · flagged`
              : `Dataset row ${payload?.[0]?.payload?.row_index ?? ''}`}
          />
          <Scatter name="Normal observations" data={normalRows} fill="#137C78" />
          <Scatter name="Flagged observations" data={anomalyRows} fill="#B5473C" />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function ExperimentCharts(props) {
  if (props.type === 'regression') {
    return <RegressionChart points={props.points} bounds={props.bounds} />;
  }
  if (props.type === 'clustering') {
    return <ClusteringChart {...props} />;
  }
  return <AnomalyChart normalRows={props.normalRows} anomalyRows={props.anomalyRows} />;
}
