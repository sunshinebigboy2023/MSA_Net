export type AnalysisFormValues = {
  text?: string;
  language?: 'auto' | 'zh' | 'en';
  enhanceTextWithTranscript?: boolean;
  video?: File | Blob;
};

export const buildAnalysisFormData = (values: AnalysisFormValues) => {
  const formData = new FormData();
  const text = (values.text || '').trim();
  const hasVideo = Boolean(values.video);

  if (text) {
    formData.append('text', text);
  }

  if (values.language && values.language !== 'auto') {
    formData.append('language', values.language);
  }

  if (text && hasVideo && values.enhanceTextWithTranscript) {
    formData.append('enhanceTextWithTranscript', 'true');
  }

  if (values.video) {
    formData.append('video', values.video);
  }

  return formData;
};
