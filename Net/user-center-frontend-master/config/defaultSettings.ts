import { Settings as LayoutSettings } from '@ant-design/pro-layout';

const Settings: LayoutSettings & {
  pwa?: boolean;
  logo?: string | false;
} = {
  navTheme: 'light',
  primaryColor: '#1677ff',
  layout: 'mix',
  contentWidth: 'Fluid',
  fixedHeader: false,
  fixSiderbar: true,
  colorWeak: false,
  title: '多模态情感分析服务',
  pwa: false,
  logo: false,
  iconfontUrl: '',
};

export default Settings;
