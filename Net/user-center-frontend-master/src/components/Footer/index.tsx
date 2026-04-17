import { DefaultFooter } from '@ant-design/pro-layout';

const Footer: React.FC = () => {
  const currentYear = new Date().getFullYear();
  return (
    <DefaultFooter
      copyright={`${currentYear} 多模态情感分析服务`}
      links={[
        {
          key: 'home',
          title: '情感分析',
          href: '/welcome',
          blankTarget: false,
        },
      ]}
    />
  );
};

export default Footer;
