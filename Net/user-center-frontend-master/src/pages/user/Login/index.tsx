import {
  CheckCircleOutlined,
  LockOutlined,
  SafetyCertificateOutlined,
  UserOutlined,
  VideoCameraOutlined,
} from '@ant-design/icons';
import { Alert, Divider, message, Space, Tabs, Typography } from 'antd';
import React, { useState } from 'react';
import { LoginForm, ProFormCheckbox, ProFormText } from '@ant-design/pro-form';
import { history, Link, useModel } from 'umi';
import Footer from '@/components/Footer';
import { login } from '@/services/ant-design-pro/api';
import styles from './index.less';

const LoginMessage: React.FC<{ content: string }> = ({ content }) => (
  <Alert style={{ marginBottom: 24 }} message={content} type="error" showIcon />
);

const BrandMark = () => <div className={styles.brandMark}>MSA</div>;

const Login: React.FC = () => {
  const [userLoginState] = useState<API.LoginResult>({});
  const [type, setType] = useState<string>('account');
  const { initialState, setInitialState } = useModel('@@initialState');

  const fetchUserInfo = async () => {
    const userInfo = await initialState?.fetchUserInfo?.();
    if (userInfo) {
      await setInitialState((s) => ({ ...s, currentUser: userInfo }));
    }
  };

  const handleSubmit = async (values: API.LoginParams) => {
    try {
      const user = await login({ ...values, type });
      if (user) {
        message.success('登录成功');
        await fetchUserInfo();

        const { query } = history.location;
        const { redirect } = query as { redirect?: string };
        history.push(redirect || '/');
      }
    } catch (error) {
      message.error('登录失败，请检查账号和密码');
    }
  };

  const { status, type: loginType } = userLoginState;
  return (
    <div className={styles.container}>
      <div className={styles.content}>
        <div className={styles.loginShell}>
          <section className={styles.brandPanel}>
            <BrandMark />
            <Typography.Title level={1}>MSA-Net</Typography.Title>
            <Typography.Paragraph>
              面向文本和视频的多模态情感分析平台，帮助你更快判断情绪倾向、模型置信度和可用模态。
            </Typography.Paragraph>
            <div className={styles.featureList}>
              <Space>
                <CheckCircleOutlined />
                中文/英文模型自动路由
              </Space>
              <Space>
                <VideoCameraOutlined />
                视频、音频、文本联合分析
              </Space>
              <Space>
                <SafetyCertificateOutlined />
                低置信度结果主动提醒
              </Space>
            </div>
          </section>

          <section className={styles.formPanel}>
            <LoginForm
              logo={false}
              title="欢迎回来"
              subTitle="登录后开始创建情感分析任务"
              initialValues={{ autoLogin: true }}
              onFinish={async (values) => {
                await handleSubmit(values as API.LoginParams);
              }}
            >
              <Tabs activeKey={type} onChange={setType}>
                <Tabs.TabPane key="account" tab="账号登录" />
              </Tabs>
              {status === 'error' && loginType === 'account' && (
                <LoginMessage content="账号或密码错误" />
              )}
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
                </>
              )}
              <div className={styles.loginActions}>
                <Space split={<Divider type="vertical" />}>
                  <ProFormCheckbox noStyle name="autoLogin">
                    自动登录
                  </ProFormCheckbox>
                  <Link to="/user/register">注册新账号</Link>
                </Space>
              </div>
            </LoginForm>
          </section>
        </div>
      </div>
      <Footer />
    </div>
  );
};

export default Login;
