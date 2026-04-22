import { buildAnalysisFormData } from './analysisForm';

const entriesOf = (formData: FormData) => Array.from(formData.entries());

describe('buildAnalysisFormData', () => {
  it('sends transcript enhancement when text and video are present', () => {
    const video = new File(['video'], 'sample.mp4', { type: 'video/mp4' });

    const formData = buildAnalysisFormData({
      text: '这段发言听起来很开心',
      language: 'zh',
      enhanceTextWithTranscript: true,
      video,
    });

    expect(entriesOf(formData)).toEqual([
      ['text', '这段发言听起来很开心'],
      ['language', 'zh'],
      ['enhanceTextWithTranscript', 'true'],
      ['video', video],
    ]);
  });

  it('does not send transcript enhancement without a video', () => {
    const formData = buildAnalysisFormData({
      text: 'I am happy',
      language: 'en',
      enhanceTextWithTranscript: true,
    });

    expect(entriesOf(formData)).toEqual([
      ['text', 'I am happy'],
      ['language', 'en'],
    ]);
  });
});
