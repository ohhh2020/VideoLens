"""单通道连接池，保留现有urllib异常协议与逐行SSE读取。"""
import io
import urllib.error
import urllib.request
from contextlib import contextmanager


@contextmanager
def _network_errors():
    import httpx
    try:
        yield
    except httpx.TimeoutException as exc:
        raise TimeoutError('云端连接池请求超时') from exc
    except httpx.RequestError as exc:
        raise urllib.error.URLError(f'{type(exc).__name__}: {exc}') from exc


class PooledTransport:
    def __init__(self):
        import httpx
        # httpx自身通过getproxies读取Windows系统代理，并保留NO_PROXY绕过规则。
        # 不传单个proxy覆盖所有协议，否则会把本机端点也强行发往代理。
        self.client = httpx.Client(follow_redirects=True, trust_env=True,
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=8, keepalive_expiry=60))

    def open(self, request, timeout):
        return _Response(self.client, request, timeout)

    def close(self):
        self.client.close()


class _Response:
    def __init__(self, client, request, timeout):
        self.client, self.request, self.timeout = client, request, timeout

    def __enter__(self):
        import httpx
        try:
            self.context = self.client.stream(self.request.get_method(), self.request.full_url,
                content=self.request.data, headers=dict(self.request.header_items()), timeout=self.timeout)
            self.response = self.context.__enter__()
            if self.response.is_error:
                body = self.response.read()
                error = urllib.error.HTTPError(self.request.full_url, self.response.status_code,
                    self.response.reason_phrase, self.response.headers, io.BytesIO(body))
                self.context.__exit__(None, None, None)
                raise error
            return self
        except httpx.TimeoutException as exc:
            raise TimeoutError('云端连接池请求超时') from exc
        except httpx.RequestError as exc:
            raise urllib.error.URLError(type(exc).__name__) from exc

    def read(self):
        with _network_errors():
            return self.response.read()

    def __iter__(self):
        with _network_errors():
            for line in self.response.iter_lines():
                yield (line + '\n').encode('utf-8')

    def __exit__(self, *args):
        return self.context.__exit__(*args)


def open_request(owner, request, timeout):
    transport = getattr(owner, '_pooled_transport', None)
    return transport.open(request, timeout) if transport else urllib.request.urlopen(request, timeout=timeout)


def enable_pool(owner):
    if getattr(owner, '_pooled_transport', None) is None:
        owner._pooled_transport = PooledTransport()


def close_pool(owner):
    transport = getattr(owner, '_pooled_transport', None)
    if transport:
        transport.close()
        owner._pooled_transport = None
