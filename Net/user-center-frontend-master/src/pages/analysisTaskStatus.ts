export type AnalysisTaskStatus =
  | 'PENDING'
  | 'QUEUED'
  | 'PREPROCESSING'
  | 'EXTRACTING'
  | 'INFERRING'
  | 'RUNNING'
  | 'RETRYING'
  | 'SUCCESS'
  | 'FAILED'
  | 'DEAD_LETTER'
  | string;

const highConcurrencyStatusPercent: Record<string, number> = {
  QUEUED: 12,
  RUNNING: 42,
  RETRYING: 68,
  DEAD_LETTER: 100,
};

const highConcurrencyStatusText: Record<string, string> = {
  QUEUED: '已进入 RabbitMQ 推理队列',
  RUNNING: 'Python worker 正在执行推理',
  RETRYING: '任务正在重试队列等待重新投递',
  DEAD_LETTER: '任务已进入死信队列',
};

const highConcurrencyStatusDescription: Record<string, string> = {
  QUEUED: '后端已完成 Redis 限流和 MySQL 持久化，任务正在等待 worker 消费。',
  RUNNING: 'worker 已拉取任务，并在固定并发池中执行 MSA-Net 特征提取和模型推理。',
  RETRYING: '上一次处理没有成功，RabbitMQ 延迟重试队列会稍后重新投递任务。',
  DEAD_LETTER: '任务超过最大重试次数，需要检查模型文件、输入视频或 worker 日志。',
};

export const isAnalysisSucceeded = (status?: string) => status === 'SUCCESS';

export const isAnalysisFailed = (status?: string) =>
  status === 'FAILED' || status === 'DEAD_LETTER';

export const isAnalysisFinished = (status?: string) =>
  isAnalysisSucceeded(status) || isAnalysisFailed(status);

export const analysisStatusPercent = (status: string | undefined, fallback = 20) =>
  (status && highConcurrencyStatusPercent[status]) || fallback;

export const analysisStatusText = (status: string | undefined, fallback?: string) =>
  (status && highConcurrencyStatusText[status]) || fallback || status || '-';

export const analysisStatusDescription = (status: string | undefined, fallback?: string) =>
  (status && highConcurrencyStatusDescription[status]) || fallback || '';
