from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import Any, TypeVar, cast

_T = TypeVar("_T")

# ZWO ASI SDK calls are blocking and are made directly on the event loop thread (see
# _run_blocking). If the camera has gone unresponsive, they can hang indefinitely, so they're
# bounded with a timeout rather than let a single dead camera freeze the whole module.
SDK_CALL_TIMEOUT = 5.0


class BlockingSdkMixin:
    """Runs blocking ASI SDK calls in daemon threads with a timeout, shared by AsiCamera and AsiVideo."""

    @staticmethod
    async def _run_blocking(
        func: Callable[[], None], timeout: float = SDK_CALL_TIMEOUT, lock: threading.Lock | None = None
    ) -> bool:
        """Run a blocking ASI SDK call in a daemon thread, so a hung call can't freeze the module.

        A plain executor isn't used here, since its worker threads are non-daemon and Python joins
        them on interpreter shutdown -- a hung call would then just move the freeze to process exit.

        Args:
            func: The blocking call to run.
            timeout: How long to wait for func to complete.
            lock: Optional lock to hold while func runs, guarding SDK access against other threads.

        Returns:
            True if func completed within timeout, False if it's still running in the background.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()

        def _set_result() -> None:
            # after a timeout, wait_for() has already cancelled the future
            if not future.done():
                future.set_result(None)

        def _wrapper() -> None:
            try:
                if lock is not None:
                    with lock:
                        func()
                else:
                    func()
            finally:
                try:
                    loop.call_soon_threadsafe(_set_result)
                except RuntimeError:
                    # event loop closed (shutdown)
                    pass

        threading.Thread(target=_wrapper, daemon=True).start()
        try:
            await asyncio.wait_for(future, timeout=timeout)
            return True
        except TimeoutError:
            return False

    async def _run_blocking_or_raise(
        self, func: Callable[[], _T], timeout: float = SDK_CALL_TIMEOUT, lock: threading.Lock | None = None
    ) -> _T:
        """Run a blocking ASI SDK call in a thread, returning its result or re-raising what it raised.

        Unlike _run_blocking(), this also carries the callable's return value/exception back to the
        caller -- several ASI calls here drive control flow via their return value or a raised
        ValueError (e.g. camera lookup by name, exposure status), which a bare fire-and-forget
        thread call would otherwise silently lose.

        Args:
            func: The blocking call to run.
            timeout: How long to wait for func to complete.
            lock: Optional lock to hold while func runs, guarding SDK access against other threads.
        """
        outcome: list[Any] = []

        def _wrapper() -> None:
            try:
                outcome.append(func())
            except BaseException as e:
                outcome.append(e)

        if not await self._run_blocking(_wrapper, timeout=timeout, lock=lock):
            raise TimeoutError(f"Timed out waiting for ASI SDK call after {timeout}s.")
        value = outcome[0]
        if isinstance(value, BaseException):
            raise value
        return cast(_T, value)


__all__ = ["BlockingSdkMixin", "SDK_CALL_TIMEOUT"]
