import { LockOutlined, SafetyCertificateOutlined, UserOutlined } from '@ant-design/icons';
import { message, Tabs } from 'antd';
import React, { useState } from 'react';
import { history } from 'umi';
import Footer from '@/components/Footer';
import { register } from '@/services/ant-design-pro/api';
import styles from './index.less';
import { LoginForm, ProFormText } from '@ant-design/pro-form';

const BrandMark = () => <div className={styles.brandMark}>MSA</div>;

const Register: React.FC = () => {
  const [type, setType] = useState<string>('account');

  const handleSubmit = async (values: API.RegisterParams) => {
    const { userPassword, checkPassword } = values;
    if (userPassword !== checkPassword) {
      message.error('两次输入的密码不一致');
      return;
    }

    try {
      const id = await register(values);
      if (id) {
        message.success('注册成功，请登录');
        const { query } = history.location;
        history.push({
          pathname: '/user/login',
          query,
        });
      }
    } catch (error) {
      message.error('注册失败，请检查输入信息');
    }
  };

  return (
    <div className={styles.container}>
      <div className={styles.content}>
        <LoginForm
          submitter={{ searchConfig: { submitText: '注册' } }}
          logo={<BrandMark />}
          title="多模态情感分析服务"
          subTitle="创建账号后即可使用文本与视频情感分析能力"
          onFinish={async (values) => {
            await handleSubmit(values as API.RegisterParams);
          }}
        >
          <Tabs activeKey={type} onChange={setType}>
            <Tabs.TabPane key="account" tab="账号注册" />
          </Tabs>
          {type === 'account' && (
            <>
              <ProFormText
                name="userAccount"
                fieldProps={{
                  size: 'large',
                  prefix: <UserOutlined className={styles.prefixIcon} />,
                }}
                placeholder="请输入账号"
                rules={[{ required: true, message: '请输入账号' }]}
              />
              <ProFormText.Password
                name="userPassword"
                fieldProps={{
                  size: 'large',
                  prefix: <LockOutlined className={styles.prefixIcon} />,
                }}
                placeholder="请输入密码"
                rules={[
                  { required: true, message: '请输入密码' },
                  { min: 8, type: 'string', message: '密码长度不能少于 8 位' },
                ]}
              />
              <ProFormText.Password
                name="checkPassword"
                fieldProps={{
                  size: 'large',
                  prefix: <LockOutlined className={styles.prefixIcon} />,
                }}
                placeholder="请再次输入密码"
                rules={[
                  { required: true, message: '请再次输入密码' },
                  { min: 8, type: 'string', message: '密码长度不能少于 8 位' },
                ]}
              />
              <ProFormText
                name="planetCode"
                fieldProps={{
                  size: 'large',
                  prefix: <SafetyCertificateOutlined className={styles.prefixIcon} />,
                }}
                placeholder="请输入注册码"
                rules={[{ required: true, message: '请输入注册码' }]}
              />
            </>
          )}
        </LoginForm>
      </div>
      <Footer />
    </div>
  );
};

export default Register;
