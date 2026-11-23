import React from 'react';
import { Link } from 'react-router-dom';
import { UploadCloud, Sparkles, BarChart2, ShieldCheck, ArrowRight } from 'lucide-react';
import Card from '../../components/common/Card';
import Button from '../../components/common/Button';
import Badge from '../../components/common/Badge';
import RetentionNotice from '../../components/common/RetentionNotice';
import ExplainerCallout from '../../components/common/ExplainerCallout';
import { useDataset } from '../../context/DatasetContext';

export default function OverviewPage() {
  const { activeDataset } = useDataset();

  const steps = [
    {
      step: '01',
      title: 'Upload & Inspect',
      description: 'Upload any CSV file up to 50 MB. We check the structure, count rows and columns, and look for formatting issues.',
      icon: UploadCloud,
    },
    {
      step: '02',
      title: 'Understand Your Data',
      description: 'We examine each column to see if it holds numbers, categories, or IDs, and suggest columns you might want to predict.',
      icon: ShieldCheck,
    },
    {
      step: '03',
      title: 'Explore Patterns',
      description: 'See how your data is distributed through simple charts, spot common values, and find relationships between columns.',
      icon: BarChart2,
    },
    {
      step: '04',
      title: 'Run ML Experiments',
      description: 'Choose what you want to learn: predict a category, estimate a number, find natural clusters, or catch unusual outliers.',
      icon: Sparkles,
    },
  ];

  return (
    <div className="space-y-10">
      {/* Editorial Hero Header */}
      <div className="border-b border-stone pb-8">
        <Badge variant="teal" className="mb-3">
          Data Analysis &amp; Machine Learning
        </Badge>
        <h1 className="font-serif text-4xl sm:text-5xl font-bold tracking-tight text-navy leading-tight">
          From raw data to intelligent decisions.
        </h1>
        <p className="mt-3 text-base sm:text-lg text-navy/70 max-w-2xl font-sans leading-relaxed">
          ModelMind helps you explore datasets, catch data quality issues, and try machine learning models—with clear explanations every step of the way.
        </p>

        <div className="mt-6 flex flex-wrap items-center gap-3">
          <Link to="/datasets">
            <Button variant="primary" size="md" icon={UploadCloud}>
              {activeDataset ? 'Go to Active Dataset' : 'Upload a Dataset'}
            </Button>
          </Link>
          <Link to="/experiments">
            <Button variant="outline" size="md" icon={ArrowRight}>
              Explore ML Experiments
            </Button>
          </Link>
        </div>
      </div>

      {/* Active Session Status / Banner */}
      {activeDataset && (
        <Card
          title="Active Dataset"
          subtitle="This file is ready for analysis and experiments."
          headerAction={
            <Link to="/datasets">
              <Button variant="outline" size="sm">
                Open Dataset Hub
              </Button>
            </Link>
          }
        >
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 font-mono text-sm">
            <div className="p-3 bg-stone-light/50 rounded-sm border border-stone/80">
              <div className="text-[11px] text-navy/60 font-sans uppercase">Filename</div>
              <div className="font-semibold text-navy truncate" title={activeDataset.filename}>
                {activeDataset.filename}
              </div>
            </div>
            <div className="p-3 bg-stone-light/50 rounded-sm border border-stone/80">
              <div className="text-[11px] text-navy/60 font-sans uppercase">Rows</div>
              <div className="font-semibold text-navy">
                {activeDataset.row_count ? activeDataset.row_count.toLocaleString() : 'N/A'}
              </div>
            </div>
            <div className="p-3 bg-stone-light/50 rounded-sm border border-stone/80">
              <div className="text-[11px] text-navy/60 font-sans uppercase">Columns</div>
              <div className="font-semibold text-navy">
                {activeDataset.column_count ?? 'N/A'}
              </div>
            </div>
            <div className="p-3 bg-stone-light/50 rounded-sm border border-stone/80">
              <div className="text-[11px] text-navy/60 font-sans uppercase">Dataset ID</div>
              <div className="font-semibold text-navy text-xs truncate" title={activeDataset.dataset_id}>
                {activeDataset.dataset_id}
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* Guided Platform Journey */}
      <div>
        <h2 className="font-serif text-2xl font-bold text-navy mb-4">
          How ModelMind Works
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {steps.map((item) => {
            const Icon = item.icon;
            return (
              <Card key={item.step} className="flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between text-xs text-navy/40 font-mono mb-3">
                    <span>{item.step}</span>
                    <Icon className="w-4 h-4 text-teal" />
                  </div>
                  <h3 className="font-serif text-base font-semibold text-navy mb-1.5">
                    {item.title}
                  </h3>
                  <p className="text-xs text-navy/70 leading-relaxed font-sans">
                    {item.description}
                  </p>
                </div>
              </Card>
            );
          })}
        </div>
      </div>

      {/* Beginner-Friendly Explainer & Retention Notice */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ExplainerCallout term="Our Approach">
          Machine learning works best when you truly understand your data first. Before training any model, take time to check for missing values, explore relationships between columns, and compare multiple models side by side.
        </ExplainerCallout>

        <RetentionNotice />
      </div>
    </div>
  );
}
