import React, { useEffect, useRef, useState } from 'react';
import { PageContainer } from '@ant-design/pro-layout';
import type { UploadFile } from 'antd/es/upload/interface';
import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Empty,
  Input,
  message,
  Progress,
  Radio,
  Row,
  Space,
  Statistic,
  Tag,
  Typography,
  Upload,
} from 'antd';
import {
  CloudUploadOutlined,
  FileTextOutlined,
  PlayCircleOutlined,
  SyncOutlined,
} from '@ant-design/icons';
import {
  getAnalysisResult,
  getAnalysisTask,
  submitAnalysis,
} from '@/services/ant-design-pro/api';
import styles from './Welcome.less';

const { Dragger } = Upload;
const { TextArea } = Input;

type LanguageMode = 'auto' | 'zh' | 'en';

const statusPercent: Record<string, number> = {
  PENDING: 15,
  PREPROCESSING: 35,
  EXTRACTING: 55,
  INFERRING: 78,
  SUCCESS: 100,
  FAILED: 100,
};

const statusText: Record<string, string> = {
  PENDING: '任务已提交',
  PREPROCESSING: '正在预处理媒体',
  EXTRACTING: '正在提取多模态特征',
  INFERRING: '正在进行情感推理',
  SUCCESS: '分析完成',
  FAILED: '分析失败',
};

const polarityText: Record<string, string> = {
  positive: '积极',
  negative: '消极',
  neutral: '中性',
};

const modalityText: Record<string, string> = {
  text: '文本',
  audio: '音频',
  video: '视频',
};

const Welcome: React.FC = () => {
  const [text, setText] = useState('');
  const [language, setLanguage] = useState<LanguageMode>('auto');
  const [videoList, setVideoList] = useState<UploadFile[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [task, setTask] = useState<API.AnalysisTask>();
  const [result, setResult] = useState<API.AnalysisResult>();
  const timerRef = useRef<number>();

  useEffect(() => {
    return () => {
      if (timerRef.current) {
        window.clearInterval(timerRef.current);
      }
    };
  }, []);

  const stopPolling = () => {
    if (timerRef.current) {
      window.clearInterval(timerRef.current);
      timerRef.current = undefined;
    }
  };

  const pollTask = (taskId: string) => {
    stopPolling();
    timerRef.current = window.setInterval(async () => {
      try {
        const current = await getAnalysisTask(taskId);
        setTask(current);
        if (current.status === 'SUCCESS') {
          stopPolling();
          const nextResult = await getAnalysisResult(taskId);
          setResult(nextResult);
          setSubmitting(false);
        }
        if (current.status === 'FAILED') {
          stopPolling();
          setSubmitting(false);
          message.error(current.error || '分析失败，请查看 MSA 服务窗口日志');
        }
      } catch (error) {
        stopPolling();
        setSubmitting(false);
        message.error('获取分析进度失败，请确认后端和 MSA 服务正在运行');
      }
    }, 1500);
  };

  const handleAnalyze = async () => {
    const content = text.trim();
    const video = videoList[0]?.originFileObj;
    if (!content && !video) {
      message.warning('请输入文本或上传视频');
      return;
    }

    const formData = new FormData();
    if (content) {
      formData.append('text', content);
    }
    if (language !== 'auto') {
      formData.append('language', language);
    }
    if (video) {
      formData.append('video', video);
    }

    try {
      stopPolling();
      setSubmitting(true);
      setResult(undefined);
      const nextTask = await submitAnalysis(formData);
      setTask(nextTask);
      pollTask(nextTask.taskId);
      message.success('分析任务已提交');
    } catch (error) {
      setSubmitting(false);
      message.error('提交分析任务失败，请检查登录状态、视频大小和服务状态');
    }
  };

  const percent = task ? statusPercent[task.status] || 20 : 0;
  const progressStatus = task?.status === 'FAILED' ? 'exception' : task?.status === 'SUCCESS' ? 'success' : 'active';
  const polarity = result?.sentimentPolarity || result?.emotionLabel || '-';

  return (
    <PageContainer
      title="多模态情感分析服务"
      content="上传文本、视频或二者组合，系统将自动提取多模态特征并返回情感分析结果。"
    >
      <div className={styles.workspace}>
        <Row gutter={[24, 24]}>
          <Col xs={24} lg={12}>
            <Card className={styles.panel} title={<Space><FileTextOutlined />分析输入</Space>}>
              <Space direction="vertical" size={20} className={styles.fullWidth}>
                <div>
                  <Typography.Text strong>文本内容</Typography.Text>
                  <TextArea
                    className={styles.textArea}
                    value={text}
                    onChange={(event) => setText(event.target.value)}
                    rows={8}
                    maxLength={2000}
                    showCount
                    placeholder="请输入需要分析的文本。文本和视频可任选其一，也可以同时提供。"
                  />
                </div>

                <div>
                  <Typography.Text strong>视频文件</Typography.Text>
                  <Dragger
                    className={styles.uploader}
                    accept="video/*,.mp4,.avi,.mov,.mkv"
                    maxCount={1}
                    fileList={videoList}
                    beforeUpload={() => false}
                    onChange={({ fileList }) => setVideoList(fileList.slice(-1))}
                    onRemove={() => {
                      setVideoList([]);
                    }}
                  >
                    <p className="ant-upload-drag-icon">
                      <CloudUploadOutlined />
                    </p>
                    <p className="ant-upload-text">点击或拖拽视频到此处</p>
                    <p className="ant-upload-hint">支持常见视频格式。较大文件上传和分析需要更长时间。</p>
                  </Dragger>
                </div>

                <div>
                  <Typography.Text strong>语言设置</Typography.Text>
                  <Radio.Group
                    className={styles.languageGroup}
                    value={language}
                    onChange={(event) => setLanguage(event.target.value)}
                    optionType="button"
                    buttonStyle="solid"
                    options={[
                      { label: '自动识别', value: 'auto' },
                      { label: '中文 SIMS', value: 'zh' },
                      { label: '英文 MOSI', value: 'en' },
                    ]}
                  />
                </div>

                <Button
                  type="primary"
                  size="large"
                  icon={<PlayCircleOutlined />}
                  loading={submitting}
                  onClick={handleAnalyze}
                  block
                >
                  开始情感分析
                </Button>
              </Space>
            </Card>
          </Col>

          <Col xs={24} lg={12}>
            <Card className={styles.panel} title={<Space><SyncOutlined />分析结果</Space>}>
              {task ? (
                <Space direction="vertical" size={20} className={styles.fullWidth}>
                  <Alert
                    type={task.status === 'FAILED' ? 'error' : task.status === 'SUCCESS' ? 'success' : 'info'}
                    showIcon
                    message={statusText[task.status] || task.status}
                    description={task.error || `任务编号：${task.taskId}`}
                  />
                  <Progress percent={percent} status={progressStatus} />

                  {result ? (
                    <>
                      <Row gutter={16}>
                        <Col span={8}>
                          <Statistic title="情感极性" value={polarityText[polarity] || polarity} />
                        </Col>
                        <Col span={8}>
                          <Statistic title="情感得分" value={result.score ?? 0} precision={3} />
                        </Col>
                        <Col span={8}>
                          <Statistic title="置信度" value={result.confidence ?? 0} precision={3} />
                        </Col>
                      </Row>

                      <Descriptions column={1} size="small" bordered>
                        <Descriptions.Item label="使用模态">
                          {(result.usedModalities || []).map((item) => (
                            <Tag color="blue" key={item}>{modalityText[item] || item}</Tag>
                          ))}
                        </Descriptions.Item>
                        <Descriptions.Item label="语言">{result.language || '-'}</Descriptions.Item>
                        <Descriptions.Item label="模型数据集">{result.modelDataset || '-'}</Descriptions.Item>
                        <Descriptions.Item label="模型条件">{result.modelCondition || '-'}</Descriptions.Item>
                        <Descriptions.Item label="处理耗时">
                          {result.processingTimeMs ? `${result.processingTimeMs} ms` : '-'}
                        </Descriptions.Item>
                        <Descriptions.Item label="音频转写">{result.transcript || '-'}</Descriptions.Item>
                      </Descriptions>
                    </>
                  ) : (
                    <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="等待分析结果" />
                  )}
                </Space>
              ) : (
                <Empty description="提交文本或视频后，分析结果会显示在这里" />
              )}
            </Card>
          </Col>
        </Row>
      </div>
    </PageContainer>
  );
};

export default Welcome;
