import queue
import threading
import time
from multiprocessing.connection import Client

from image.image_data_extractor import image_data_extractor
from utils.config import config
from utils.constants import ImageGenerationType
from utils.logging_setup import get_logger
from utils.translations import _
logger = get_logger("sd_runner_client")

class SDRunnerClient:
    COMMAND_CLOSE_SERVER = 'close server'
    COMMAND_CLOSE_CONNECTION = 'close connection'
    COMMAND_VALIDATE = 'validate'

    def __init__(self, host='localhost', port=config.sd_runner_client_port):
        self._host = host
        self._port = port
        self._conn = None
        self._request_queue = queue.Queue()
        self._worker_thread = None
        self._shutdown = False
        self._worker_lock = threading.Lock()

    def start(self, max_retries: int = 1, retry_delay: float = 0.5) -> None:
        """Connect to the SD runner server, retrying on ConnectionRefusedError.

        Retries handle the case where the server is still starting up or is
        briefly unavailable after a crash.  Non-connection errors are raised
        immediately without retrying.
        """
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                self._conn = Client(
                    (self._host, self._port),
                    authkey=str.encode(config.sd_runner_client_password),
                )
                logger.info("Started SDRunner Client")
                return
            except ConnectionRefusedError as e:
                last_error = e
                if attempt < max_retries - 1:
                    logger.warning(
                        f"SD Runner connection refused (attempt {attempt + 1}/{max_retries}),"
                        f" retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
            except Exception as e:
                logger.error(f"Failed to connect to SD Runner: {e}")
                raise
        logger.error(f"Failed to connect to SD Runner after {max_retries} attempts: {last_error}")
        raise last_error

    def send(self, msg):
        if config.debug:
            logger.debug(f"Sending {msg} to SD Runner")
        self._conn.send(msg)
        return self._conn.recv()

    def close(self):
        try:
            self._conn.send(SDRunnerClient.COMMAND_CLOSE_CONNECTION)
            self._conn.close()
            self._conn = None
            logger.info("Closed SD Runner Client")
        except Exception as e:
            logger.error(f"Failed to close SD Runner Client: {e}")
            raise e

    def validate_connection(self):
        try:
            resp = self.send(SDRunnerClient.COMMAND_VALIDATE)
            if resp != 'valid':
                self.close()
                raise Exception(f'SD Runner connection failed validation: {resp}')
        except Exception as e:
            self.close()
            raise Exception(f'Failed to connect to SD Runner: {e}')

    def validate_image_for_type(self, _type, base_image):
        if _type == ImageGenerationType.REDO_PROMPT:
            prompt, software_type = image_data_extractor.extract_prompt(base_image)
            if prompt is None:
                self.close()
                raise Exception(_('Image does not contain a prompt to redo!'))

    def _start_worker(self):
        """Start the worker thread that processes queued requests (lazy initialization)."""
        with self._worker_lock:
            if self._worker_thread is None or not self._worker_thread.is_alive():
                self._shutdown = False
                self._worker_thread = threading.Thread(target=self._process_queue, daemon=True)
                self._worker_thread.start()

    def _process_queue(self):
        """Worker thread that processes requests from the queue one at a time."""
        while not self._shutdown:
            try:
                # Get request from queue with timeout to allow checking shutdown flag
                request = self._request_queue.get(timeout=1.0)
                if request is None:  # Shutdown signal
                    break
                
                request_type, args, result_container, condition = request
                
                try:
                    if request_type == 'run':
                        result = self._run_internal(*args)
                        result_container['result'] = result
                    elif request_type == 'run_on_directory':
                        result = self._run_on_directory_internal(*args)
                        result_container['result'] = result
                except Exception as e:
                    result_container['exception'] = e
                
                result_container['done'] = True
                with condition:
                    condition.notify()
                
                self._request_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in worker thread: {e}")

    def _run_internal(
        self,
        _type,
        base_image,
        append=False,
        positive_prompt=None,
        negative_prompt=None,
        edit_suffix=None,
    ):
        """Internal method that performs the actual run operation."""
        if not isinstance(_type, ImageGenerationType):
            raise TypeError(f'{_type} is not a valid ImageGenerationType')
        self.start()
        self.validate_image_for_type(_type, base_image)
        self.validate_connection()
        try:
            args = {'image': base_image, 'append': append}
            if positive_prompt is not None:
                args['positive_prompt'] = positive_prompt
            if negative_prompt is not None:
                args['negative_prompt'] = negative_prompt
            if edit_suffix is not None:
                args['edit_suffix'] = edit_suffix
            command  = {'command': 'run', 'type': _type.value, 'args': args}
            resp = self.send(command)
            if "error" in resp:
                self.close()
                raise Exception(f'SD Runner failed to start run {_type} on file {base_image}\n{resp["error"]}: {resp["data"]}')
            logger.info(f"SD Runner started run {_type} on file {base_image}")
            self.close()
            return resp['data'] if "data" in resp else None
        except Exception as e:
            try:
                self.close()
            except Exception as e2:
               pass
            raise Exception(f'Failed to start run on SD Runner: {e}')

    def run(
        self,
        _type,
        base_image,
        append=False,
        positive_prompt=None,
        negative_prompt=None,
        edit_suffix=None,
    ):
        """Queue a run request and wait for it to complete."""
        self._start_worker()  # Lazy initialization
        result_container = {'done': False, 'result': None, 'exception': None}
        condition = threading.Condition()
        self._request_queue.put(
            (
                'run',
                (_type, base_image, append, positive_prompt, negative_prompt, edit_suffix),
                result_container,
                condition,
            )
        )
        
        # Wait for completion
        with condition:
            while not result_container['done']:
                condition.wait()
        
        if result_container['exception'] is not None:
            raise result_container['exception']
        
        return result_container['result']

    def _run_on_directory_internal(self, _type, directory_path, append=False):
        """Internal method that performs the actual run_on_directory operation."""
        if not isinstance(_type, ImageGenerationType):
            raise TypeError(f'{_type} is not a valid ImageGenerationType')
        self.start()
        try:
            # Skip image validation as requested - directly send command
            self.validate_connection()
            command = {'command': 'run', 'type': _type.value,
                      'args': {'image': directory_path, 'append': append}}
            resp = self.send(command)
            if "error" in resp:
                self.close()
                raise Exception(f'SD Runner failed to start run {_type} on directory {directory_path}\n{resp["error"]}: {resp["data"]}')
            logger.info(f"SD Runner started run {_type} on directory {directory_path}")
            self.close()
            return resp['data'] if "data" in resp else None
        except Exception as e:
            try:
                self.close()
            except Exception:
                pass
            raise Exception(f'Failed to start run on SD Runner: {e}')

    def run_on_directory(self, _type, directory_path, append=False):
        """
        Run image generation on a directory path, skipping image validation.
        Queue the request and wait for it to complete.
        """
        self._start_worker()  # Lazy initialization
        result_container = {'done': False, 'result': None, 'exception': None}
        condition = threading.Condition()
        self._request_queue.put(('run_on_directory', (_type, directory_path, append), result_container, condition))
        
        # Wait for completion
        with condition:
            while not result_container['done']:
                condition.wait()
        
        if result_container['exception'] is not None:
            raise result_container['exception']
        
        return result_container['result']

    def run_batch(self, _type, requests: list[dict]) -> int:
        """Send all requests as a single blob; the server enqueues and acks immediately.

        *requests* is a list of args dicts (must include at least ``image``).
        All requests share the same generation type.
        Returns the number of items the server confirmed as queued.
        """
        if not requests:
            return 0
        if not isinstance(_type, ImageGenerationType):
            raise TypeError(f'{_type} is not a valid ImageGenerationType')
        self.start()
        try:
            self.validate_connection()
            batch = [
                {'type': _type.value, 'args': args}
                for args in requests
            ]
            resp = self.send({'command': 'run_batch', 'requests': batch})
            if 'error' in resp:
                raise Exception(
                    f'SD Runner batch error: {resp["error"]}: {resp.get("data")}'
                )
            count = resp.get('count', len(requests))
            if config.debug:
                from collections import defaultdict
                by_image = defaultdict(list)
                for args in requests:
                    by_image[args.get('image', '?')].append(args.get('edit_suffix', ''))
                lines = "\n".join(
                    f"  {img}  [{', '.join(s for s in suffixes if s)}]"
                    for img, suffixes in by_image.items()
                )
                logger.info("SD Runner queued %d %s request(s):\n%s", count, _type.value, lines)
            else:
                logger.info("SD Runner queued %d batch generation request(s)", count)
            return count
        finally:
            self.close()

    def stop(self):
        """Stop the worker thread and close the connection."""
        self._shutdown = True
        self._request_queue.put(None)  # Signal shutdown
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)
        if self._conn is not None:
            try:
                self._conn.send(SDRunnerClient.COMMAND_CLOSE_SERVER)
                self._conn.close()
            except Exception:
                pass

