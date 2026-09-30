
from cpython cimport pythread

cdef extern from *:
    """
    typedef unsigned long pythread_t;

    /* Without the GIL, the lock owner is read by threads that do not hold
       the lock, so it must be accessed atomically. */
    #ifdef Py_GIL_DISABLED
    #define __lupa_FREE_THREADED 1
    #define __lupa_load_owner(p) _Py_atomic_load_ulong_relaxed(p)
    #define __lupa_store_owner(p, v) _Py_atomic_store_ulong_relaxed(p, v)
    #else
    #define __lupa_FREE_THREADED 0
    #define __lupa_load_owner(p) (*(p))
    #define __lupa_store_owner(p, v) (*(p) = (v))
    #endif
    """
    ctypedef unsigned long pythread_t
    const bint FREE_THREADED "__lupa_FREE_THREADED"
    pythread_t load_owner "__lupa_load_owner" (pythread_t* owner) noexcept nogil
    void store_owner "__lupa_store_owner" (pythread_t* owner, pythread_t value) noexcept nogil


cdef class FastRLock:
    """Fast, re-entrant locking.

    Under uncongested conditions, the lock is never acquired but only
    counted.  Only when a second thread comes in and notices that the
    lock is needed, it acquires the lock and notifies the first thread
    to release it when it's done.  This is all made possible by the
    wonderful GIL.

    Without the GIL (free-threaded CPython), there is nothing that makes
    the counting safe, so the real lock is always acquired instead.
    """
    cdef pythread.PyThread_type_lock _real_lock
    cdef pythread_t _owner               # ID of thread owning the lock
    cdef unsigned int _count             # re-entry count
    cdef unsigned int _pending_requests  # number of pending requests for real lock
    cdef bint _is_locked                 # whether the real lock is acquired

    def __cinit__(self):
        self._owner = 0
        self._count = 0
        self._is_locked = False
        self._pending_requests = 0
        self._real_lock = pythread.PyThread_allocate_lock()
        if self._real_lock is NULL:
            raise MemoryError

    def __dealloc__(self):
        if self._real_lock is not NULL:
            pythread.PyThread_free_lock(self._real_lock)
            self._real_lock = NULL

    def acquire(self, bint blocking=True):
        return lock_lock(self, pythread.PyThread_get_thread_ident(), blocking)

    def release(self):
        if not _is_owned_by(self, pythread.PyThread_get_thread_ident()):
            raise RuntimeError("cannot release un-acquired lock")
        unlock_lock(self)

    # compatibility with RLock

    def __enter__(self):
        # self.acquire()
        return lock_lock(self, pythread.PyThread_get_thread_ident(), True)

    def __exit__(self, t, v, tb):
        # self.release()
        if not _is_owned_by(self, pythread.PyThread_get_thread_ident()):
            raise RuntimeError("cannot release un-acquired lock")
        unlock_lock(self)

    def _is_owned(self):
        return _is_owned_by(self, pythread.PyThread_get_thread_ident())


cdef inline bint _is_owned_by(FastRLock lock, pythread_t current_thread) noexcept nogil:
    # Only the owning thread writes the count, so it is safe to read
    # once the owner is known to be the current thread.
    return load_owner(&lock._owner) == current_thread and lock._count > 0


cdef inline bint lock_lock(FastRLock lock, pythread_t current_thread, bint blocking) noexcept nogil:
    # Note that this function *must* hold the GIL when being called.
    # We just use 'nogil' in the signature to make sure that no Python
    # code execution slips in that might free the GIL

    if FREE_THREADED:
        return _lock_lock_free_threaded(lock, current_thread, blocking)

    if lock._count:
        # locked! - by myself?
        if current_thread == lock._owner:
            lock._count += 1
            return 1
    elif lock._pending_requests == 0:
        # not locked, not requested - go!
        lock._owner = current_thread
        lock._count = 1
        return 1
    # need to get the real lock
    return _acquire_lock(
        lock, current_thread,
        pythread.WAIT_LOCK if blocking else pythread.NOWAIT_LOCK)

cdef bint _lock_lock_free_threaded(FastRLock lock, pythread_t current_thread, bint blocking) noexcept nogil:
    # Without the GIL, the fast path of counting without locking is racy.
    # Always take the real lock, and only re-enter it without locking
    # when this thread already owns it.
    if load_owner(&lock._owner) == current_thread:
        lock._count += 1
        return 1
    if not pythread.PyThread_acquire_lock(lock._real_lock, pythread.NOWAIT_LOCK):
        if not blocking:
            return 0
        # Detach from the interpreter while waiting, so that a
        # stop-the-world pause (e.g. the garbage collector) is not
        # blocked by this thread.
        with nogil:
            pythread.PyThread_acquire_lock(lock._real_lock, pythread.WAIT_LOCK)
    lock._count = 1
    store_owner(&lock._owner, current_thread)
    return 1


cdef bint _acquire_lock(FastRLock lock, pythread_t current_thread, int wait) noexcept nogil:
    # Note that this function *must* hold the GIL when being called.
    # We just use 'nogil' in the signature to make sure that no Python
    # code execution slips in that might free the GIL

    if not lock._is_locked and lock._pending_requests == 0:
        # someone owns it but didn't acquire the real lock - do that
        # now and tell the owner to release it when done. Note that we
        # do not release the GIL here as we must absolutely be the one
        # who acquires the lock now.
        if not pythread.PyThread_acquire_lock(lock._real_lock, wait):
            return 0
        #assert not lock._is_locked
        lock._is_locked = True
    lock._pending_requests += 1
    with nogil:
        # wait for the lock owning thread to release it
        locked = pythread.PyThread_acquire_lock(lock._real_lock, wait)
    lock._pending_requests -= 1
    #assert not lock._is_locked
    #assert lock._count == 0
    if not locked:
        return 0
    lock._is_locked = True
    lock._owner = current_thread
    lock._count = 1
    return 1

cdef inline void unlock_lock(FastRLock lock) noexcept nogil:
    # Note that this function *must* hold the GIL when being called.
    # We just use 'nogil' in the signature to make sure that no Python
    # code execution slips in that might free the GIL

    #assert lock._owner == pythread.PyThread_get_thread_ident()
    #assert lock._count > 0
    lock._count -= 1
    if FREE_THREADED:
        if lock._count == 0:
            store_owner(&lock._owner, 0)
            pythread.PyThread_release_lock(lock._real_lock)
        return
    if lock._count == 0:
        if lock._is_locked:
            pythread.PyThread_release_lock(lock._real_lock)
            lock._is_locked = False
