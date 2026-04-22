import {
  analysisStatusDescription,
  analysisStatusPercent,
  analysisStatusText,
  isAnalysisFailed,
  isAnalysisFinished,
  isAnalysisSucceeded,
} from './analysisTaskStatus';

describe('analysisTaskStatus', () => {
  it('maps high concurrency queue states to visible progress', () => {
    expect(analysisStatusPercent('QUEUED')).toBe(12);
    expect(analysisStatusPercent('RUNNING')).toBe(42);
    expect(analysisStatusPercent('RETRYING')).toBe(68);
    expect(analysisStatusPercent('DEAD_LETTER')).toBe(100);
  });

  it('marks success and dead-letter states as terminal', () => {
    expect(isAnalysisSucceeded('SUCCESS')).toBe(true);
    expect(isAnalysisFinished('SUCCESS')).toBe(true);
    expect(isAnalysisFailed('DEAD_LETTER')).toBe(true);
    expect(isAnalysisFinished('DEAD_LETTER')).toBe(true);
    expect(isAnalysisFinished('RUNNING')).toBe(false);
  });

  it('provides readable labels for resume-demo architecture states', () => {
    expect(analysisStatusText('QUEUED')).toContain('RabbitMQ');
    expect(analysisStatusText('RUNNING')).toContain('worker');
    expect(analysisStatusDescription('QUEUED')).toContain('Redis');
    expect(analysisStatusDescription('RUNNING')).toContain('MSA-Net');
  });
});
