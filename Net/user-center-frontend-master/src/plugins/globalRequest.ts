import { message } from 'antd';
import { stringify } from 'querystring';
import { extend } from 'umi-request';
import { history } from '@@/core/history';

const request = extend({
  credentials: 'include',
});

request.interceptors.request.use((url, options): any => {
  return {
    url,
    options: {
      ...options,
      headers: options.headers,
    },
  };
});

request.interceptors.response.use(async (response): Promise<any> => {
  const res = await response.clone().json();
  if (res.code === 0) {
    return res.data;
  }
  if (res.code === 40100) {
    history.replace({
      pathname: '/user/login',
      search: stringify({
        redirect: location.pathname,
      }),
    });
  } else {
    message.error(res.description || res.message || '请求失败');
  }
  return res.data;
});

export default request;
