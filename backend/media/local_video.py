"""本机文件选择与零复制附件：只创建文件软链接，不上传原视频。"""
from __future__ import annotations

import ctypes
from pathlib import Path
import os
import threading
import uuid

from fastapi import HTTPException
from backend.media.ffmpeg_utils import probe_media
from backend.media.parser import DEFAULT_ALLOWED_EXTENSIONS

_picker_lock = threading.Lock()


def select_native_video():
    """由用户点击网页按钮后弹出系统文件选择器，不改变工作目录。"""
    if os.name != 'nt':
        raise HTTPException(501, '当前系统请使用完整路径选择本地视频')
    if not _picker_lock.acquire(blocking=False):
        raise HTTPException(409, '本机文件选择窗口已打开')
    try:
        from ctypes import wintypes as w
        class OpenFileName(ctypes.Structure):
            _fields_ = [('lStructSize',w.DWORD),('hwndOwner',w.HWND),('hInstance',w.HINSTANCE),
                ('lpstrFilter',w.LPCWSTR),('lpstrCustomFilter',w.LPWSTR),('nMaxCustFilter',w.DWORD),
                ('nFilterIndex',w.DWORD),('lpstrFile',w.LPWSTR),('nMaxFile',w.DWORD),
                ('lpstrFileTitle',w.LPWSTR),('nMaxFileTitle',w.DWORD),('lpstrInitialDir',w.LPCWSTR),
                ('lpstrTitle',w.LPCWSTR),('Flags',w.DWORD),('nFileOffset',w.WORD),('nFileExtension',w.WORD),
                ('lpstrDefExt',w.LPCWSTR),('lCustData',w.LPARAM),('lpfnHook',ctypes.c_void_p),
                ('lpTemplateName',w.LPCWSTR),('pvReserved',ctypes.c_void_p),('dwReserved',w.DWORD),('FlagsEx',w.DWORD)]
        dialog=ctypes.WinDLL('comdlg32',use_last_error=True)
        dialog.GetOpenFileNameW.argtypes=[ctypes.POINTER(OpenFileName)]
        dialog.GetOpenFileNameW.restype=w.BOOL
        dialog.CommDlgExtendedError.restype=w.DWORD
        user=ctypes.WinDLL('user32',use_last_error=True)
        user.GetForegroundWindow.restype=w.HWND
        buffer=ctypes.create_unicode_buffer(32768)
        options=OpenFileName();options.lStructSize=ctypes.sizeof(options)
        options.hwndOwner=user.GetForegroundWindow()
        extensions=';'.join('*.'+ext for ext in sorted(DEFAULT_ALLOWED_EXTENSIONS))
        options.lpstrFilter='视频文件\0'+extensions+'\0\0'
        options.lpstrFile=ctypes.cast(buffer,w.LPWSTR);options.nMaxFile=len(buffer)
        options.lpstrTitle='选择本地视频（使用原文件，不复制）'
        options.Flags=0x80000|0x1000|0x800|0x8|0x2000000
        if not dialog.GetOpenFileNameW(ctypes.byref(options)):
            error=dialog.CommDlgExtendedError()
            if error:
                raise HTTPException(500,f'无法打开本机文件选择窗口（{error}）')
            return None
        return buffer.value
    finally:
        _picker_lock.release()


def describe_local_file(raw, config):
    text=str(raw).strip().strip('"')
    candidate=Path(text).expanduser()
    if not candidate.is_absolute():
        raise HTTPException(422,'请输入视频的完整本地路径')
    path=candidate.resolve()
    if not path.is_file():
        raise HTTPException(404,'本地视频不存在或已被移动')
    if path.suffix.lower().lstrip('.') not in DEFAULT_ALLOWED_EXTENSIONS:
        raise HTTPException(415,'请选择支持的视频文件格式')
    stat=path.stat()
    if stat.st_size<=0:
        raise HTTPException(415,'视频文件为空')
    if stat.st_size>config.limits.max_file_bytes:
        raise HTTPException(413,'视频超过文件大小限制')
    return {'path':str(path),'filename':path.name,'size_bytes':stat.st_size,
            'last_modified':stat.st_mtime_ns//1_000_000,'local':True}


def attach_local_video(service,cid,raw,*,expected_size=None,expected_modified=None):
    """发送时验证媒体并登记软链接；路径只来自用户明确选择，不经过模型生成。"""
    conversation=service.store.get(cid)
    if conversation['video_id'] or service.store.active(cid):
        raise HTTPException(409,'当前会话已绑定视频或正在处理，请新建会话')
    item=describe_local_file(raw,service.config)
    if (expected_size is not None and item['size_bytes']!=expected_size) or (
            expected_modified is not None and item['last_modified']!=expected_modified):
        raise HTTPException(409,'原文件已发生变化，请重新选择后发送')
    source=Path(item['path'])
    info=probe_media(source)
    if not info.has_video or info.width<=0 or info.height<=0:
        raise HTTPException(415,'文件没有有效视频画面')
    if info.duration_ms>service.config.limits.max_duration_ms:
        raise HTTPException(413,'视频时长超过限制')
    directory=Path(service.config.path('data'))/'uploads';directory.mkdir(parents=True,exist_ok=True)
    aid='u'+uuid.uuid4().hex
    link=directory/('local-'+aid+source.suffix.lower())
    try:
        link.symlink_to(source)
        service.store.add_attachment(cid,aid,item['filename'],link,item['size_bytes'])
    except BaseException as exc:
        if link.is_symlink():
            link.unlink(missing_ok=True)
        if isinstance(exc,OSError):
            raise HTTPException(507,'无法创建本地视频软链接，请检查目录写入权限及Windows符号链接权限；原文件未被复制') from exc
        raise
    return {'id':aid,**item,'link_path':str(link)}
