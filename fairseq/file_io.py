#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import shutil
from typing import List, Optional
import json, io, torch 
from petrel_client.client import Client

logger = logging.getLogger(__file__)





class CephManager:

    def __init__(self, s2_conf_path='~/petreloss.conf'):
        self.conf_path = s2_conf_path
        self._client = Client(conf_path=s2_conf_path)

    def readlines(self, url):

        response = self._client.get(url, enable_stream=True, no_cache=True)

        lines = []
        for line in response.iter_lines():
            lines.append(line.decode('utf-8'))
        return lines

    def load_data(self, path, ceph_read=False):
        if ceph_read:
            return self.readlines(path)
        else:
            return self._client.get(path)

    def get(self, file_path):
        return self._client.get(file_path)


    def load_json(self, json_url):
        return json.loads(self.load_data(json_url, ceph_read=False))

    def load_model(self, model_path, map_location):
        file_bytes = self._client.get(model_path)
        buffer = io.BytesIO(file_bytes)
        return torch.load(buffer, map_location=map_location)

    def write(self, save_dir, obj):
        self._client.put(save_dir, obj)

    def put_text(self,
                 obj: str,
                 filepath,
                 encoding: str = 'utf-8') -> None:
        self.write(filepath, bytes(obj, encoding=encoding))

    def exists(self, url):
        return self._client.contains(url)
    
    def remove(self, url):
        return self._client.delete(url)
    
    def isdir(self, url):
        return self._client.isdir(url)

    def isfile(self, url):
        return self.exists(url) and not self.isdir(url)

    def listdir(self, url):
        return self._client.list(url)

    def copy(self, src_path, dst_path, overwrite):
        if not overwrite and self.exists(dst_path):
            pass
        object = self._client.get(src_path)
        self._client.put(dst_path, object)
        return dst_path
try:
    ceph_manager = CephManager()
except:
    logging.warning("ceph manager has not be registered")

try:
    from iopath.common.file_io import g_pathmgr as IOPathManager

    try:
        # [FB only - for now] AWS PathHandler for PathManager
        from .fb_pathhandlers import S3PathHandler

        IOPathManager.register_handler(S3PathHandler())
    except KeyError:
        logging.warning("S3PathHandler already registered.")
    except ImportError:
        logging.debug(
            "S3PathHandler couldn't be imported. Either missing fb-only files, or boto3 module."
        )

except ImportError:
    IOPathManager = None


def _is_use_ceph(path):
    return True if "s3://" in path else False

class PathManager:
    """
    Wrapper for insulating OSS I/O (using Python builtin operations) from
    iopath's PathManager abstraction (for transparently handling various
    internal backends).
    """

    @staticmethod
    def open(
        path: str,
        mode: str = "r",
        buffering: int = -1,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
    ):
        if IOPathManager:
            return IOPathManager.open(
                path=path,
                mode=mode,
                buffering=buffering,
                encoding=encoding,
                errors=errors,
                newline=newline,
            )
        elif _is_use_ceph(path):
            return path
        return open(
            path,
            mode=mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )

    @staticmethod
    def copy(src_path: str, dst_path: str, overwrite: bool = False) -> bool:
        if IOPathManager:
            return IOPathManager.copy(
                src_path=src_path, dst_path=dst_path, overwrite=overwrite
            )
        elif _is_use_ceph(src_path):
            return ceph_manager.copy(src_path, dst_path, overwrite)
        return shutil.copyfile(src_path, dst_path)

    @staticmethod
    def get_local_path(path: str, **kwargs) -> str:
        if IOPathManager:
            return IOPathManager.get_local_path(path, **kwargs)
        return path

    @staticmethod
    def get_ceph_manager():
        return ceph_manager

    @staticmethod
    def exists(path: str) -> bool:
        if IOPathManager:
            return IOPathManager.exists(path)
        elif _is_use_ceph(path):
            return ceph_manager.exists(path)
        return os.path.exists(path)

    @staticmethod
    def isfile(path: str) -> bool:
        if IOPathManager:
            return IOPathManager.isfile(path)
        elif _is_use_ceph(path):
            return ceph_manager.isfile(path)
        return os.path.isfile(path)

    @staticmethod
    def ls(path: str) -> List[str]:
        if IOPathManager:
            return IOPathManager.ls(path)
        elif _is_use_ceph(path):
            return ceph_manager.listdir(path)
        return os.listdir(path)

    @staticmethod
    def mkdirs(path: str) -> None:
        if IOPathManager:
            return IOPathManager.mkdirs(path)
        elif _is_use_ceph(path):
            return
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def rm(path: str) -> None:
        if IOPathManager:
            return IOPathManager.rm(path)
        elif _is_use_ceph(path):
            return ceph_manager.remove(path)
        else:
            os.remove(path)

    @staticmethod
    def chmod(path: str, mode: int) -> None:
        if not PathManager.path_requires_pathmanager(path):
            os.chmod(path, mode)

    @staticmethod
    def register_handler(handler) -> None:
        if IOPathManager:
            return IOPathManager.register_handler(handler=handler)

    @staticmethod
    def copy_from_local(
        local_path: str, dst_path: str, overwrite: bool = False, **kwargs
    ) -> None:
        if IOPathManager:
            return IOPathManager.copy_from_local(
                local_path=local_path, dst_path=dst_path, overwrite=overwrite, **kwargs
            )
        elif _is_use_ceph(local_path):
            return ceph_manager.copy(local_path, dst_path, overwrite)
        return shutil.copyfile(local_path, dst_path)

    @staticmethod
    def path_requires_pathmanager(path: str) -> bool:
        """Do we require PathManager to access given path?"""
        if IOPathManager:
            for p in IOPathManager._path_handlers.keys():
                if path.startswith(p):
                    return True
        elif _is_use_ceph(path):
            return True
        return False

    @staticmethod
    def supports_rename(path: str) -> bool:
        # PathManager doesn't yet support renames
        return not PathManager.path_requires_pathmanager(path)

    @staticmethod
    def rename(src: str, dst: str):
        if "s3://" in src:
            os.system("aws s3 mv %s %s" % (src, dst))
        else:
            os.rename(src, dst)

    """
    ioPath async PathManager methods:
    """

    @staticmethod
    def opena(
        path: str,
        mode: str = "r",
        buffering: int = -1,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
    ):
        """
        Return file descriptor with asynchronous write operations.
        """
        global IOPathManager
        if not IOPathManager:
            logging.info("ioPath is initializing PathManager.")
            try:
                from iopath.common.file_io import PathManager

                IOPathManager = PathManager()
            except Exception:
                logging.exception("Failed to initialize ioPath PathManager object.")
        return IOPathManager.opena(
            path=path,
            mode=mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )

    @staticmethod
    def async_close() -> bool:
        """
        Wait for files to be written and clean up asynchronous PathManager.
        NOTE: `PathManager.async_close()` must be called at the end of any
        script that uses `PathManager.opena(...)`.
        """
        global IOPathManager
        if IOPathManager:
            return IOPathManager.async_close()
        return False



class CEPHFileUtil(object):
    def __init__(self, s3cfg_path='~/petreloss.conf'):
        self.ceph_handler = CephManager(s3cfg_path)

    @staticmethod
    def _is_use_ceph(path):
        return True if "s3://" in path else False

    def make_dirs(self, dir_path, exist_ok):

        if not self._is_use_ceph(dir_path):
            os.makedirs(dir_path, exist_ok=exist_ok)

    def exists(self, file_path):

        use_ceph = self._is_use_ceph(file_path)
        return self.ceph_handler.exists(file_path) if use_ceph else os.path.exists(file_path)

    def lexists(self, file_path):

        def lexists_for_ceph(file_path):
            tmp_str = file_path[:-1] if file_path.endswith("/") else file_path
            return self.exists(tmp_str[:tmp_str.rindex("/")])

        use_ceph = self._is_use_ceph(file_path)
        return lexists_for_ceph(file_path) if use_ceph else os.path.lexists(file_path)

    def remove(self, file_path):

        use_ceph = self._is_use_ceph(file_path)
        self.ceph_handler.remove(file_path) if use_ceph else os.remove(file_path)

    def load_checkpoint(self, file_path, map_location):
        def _load_from_local(local_path, m_location):
            with open(local_path, "rb") as f:
                state = torch.load(f, map_location=m_location)
            return state

        def _load_from_ceph(url, m_location):
            return self.ceph_handler.load_model(url, map_location=m_location)

        use_ceph = self._is_use_ceph(file_path)
        return _load_from_ceph(file_path, map_location) if use_ceph else _load_from_local(file_path, map_location)

    def readlines(self, url):
        return self.ceph_handler.readlines(url)

    def get(self, url):
        return self.ceph_handler.get(url)

    def put(self, url, obj):
        self.ceph_handler.write(url, obj)