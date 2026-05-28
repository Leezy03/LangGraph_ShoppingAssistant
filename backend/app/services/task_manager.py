"""In-memory task status and trace management for shopping analysis."""

from __future__ import annotations

from datetime import datetime, timezone
from threading import RLock
from typing import Dict, Optional
from uuid import uuid4

from ..models.schemas import ShoppingAnalysisTaskStatus, ShoppingReport, TaskTraceEvent


def _now() -> datetime:
    return datetime.now(timezone.utc)


class InMemoryTaskStore:
    """Thread-safe in-memory task store.

    This is intentionally process-local. It gives the app real task status and
    trace visibility for development/demo usage; production deployment should
    back this with Redis/Postgres if multiple workers are used.
    """

    def __init__(self):
        self._tasks: Dict[str, ShoppingAnalysisTaskStatus] = {}
        self._lock = RLock()

    def create_task(self) -> str:
        task_id = str(uuid4())
        now = _now()
        with self._lock:
            self._tasks[task_id] = ShoppingAnalysisTaskStatus(
                task_id=task_id,
                status="pending",
                current_step=None,
                progress=0,
                message="任务已创建",
                created_at=now,
                updated_at=now,
            )
        return task_id

    def get_task(self, task_id: str) -> Optional[ShoppingAnalysisTaskStatus]:
        with self._lock:
            task = self._tasks.get(task_id)
            return task.model_copy(deep=True) if task else None

    def mark_running(self, task_id: str, message: str = "任务运行中"):
        with self._lock:
            task = self._require_task(task_id)
            task.status = "running"
            task.message = message
            task.updated_at = _now()

    def start_step(self, task_id: str, step_key: str, step_name: str, progress: int, message: str) -> str:
        event_id = str(uuid4())
        now = _now()
        with self._lock:
            task = self._require_task(task_id)
            task.status = "running"
            task.current_step = step_name
            task.progress = max(task.progress, min(progress, 99))
            task.message = message
            task.updated_at = now
            task.trace.append(
                TaskTraceEvent(
                    event_id=event_id,
                    step_key=step_key,
                    step_name=step_name,
                    status="running",
                    message=message,
                    started_at=now,
                )
            )
        return event_id

    def finish_step(
        self,
        task_id: str,
        event_id: str,
        status: str,
        message: str,
        progress: int,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
    ):
        now = _now()
        with self._lock:
            task = self._require_task(task_id)
            event = self._find_event(task, event_id)
            event.status = status
            event.message = message
            event.ended_at = now
            event.duration_ms = int((now - event.started_at).total_seconds() * 1000)
            event.error_type = error_type
            event.error_message = error_message
            task.progress = max(task.progress, min(progress, 99))
            task.message = message
            task.updated_at = now

    def complete_task(self, task_id: str, report: ShoppingReport, status: str = "succeeded"):
        now = _now()
        with self._lock:
            task = self._require_task(task_id)
            task.status = status
            task.current_step = None
            task.progress = 100
            task.message = "分析完成" if status == "succeeded" else "分析完成，存在降级结果"
            task.report = report
            task.completed_at = now
            task.updated_at = now

    def fail_task(self, task_id: str, error: str):
        now = _now()
        with self._lock:
            task = self._require_task(task_id)
            task.status = "failed"
            task.current_step = None
            task.message = "分析失败"
            task.error = error
            task.completed_at = now
            task.updated_at = now

    def clear(self):
        with self._lock:
            self._tasks.clear()

    def _require_task(self, task_id: str) -> ShoppingAnalysisTaskStatus:
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"任务不存在: {task_id}")
        return task

    def _find_event(self, task: ShoppingAnalysisTaskStatus, event_id: str) -> TaskTraceEvent:
        for event in task.trace:
            if event.event_id == event_id:
                return event
        raise KeyError(f"Trace事件不存在: {event_id}")


class TaskTracer:
    """Small adapter used by LangGraph nodes to update task status and trace."""

    _start_progress = {
        "candidate": 5,
        "review": 25,
        "price": 25,
        "red_flag": 25,
        "report": 80,
    }
    _end_progress = {
        "candidate": 20,
        "review": 38,
        "price": 56,
        "red_flag": 74,
        "report": 95,
    }

    def __init__(self, task_id: str, store: InMemoryTaskStore):
        self.task_id = task_id
        self.store = store
        self._lock = RLock()
        self._degraded = False

    def start_step(self, step_key: str, step_name: str) -> str:
        return self.store.start_step(
            task_id=self.task_id,
            step_key=step_key,
            step_name=step_name,
            progress=self._start_progress.get(step_key, 10),
            message=f"{step_name}开始执行",
        )

    def finish_step(
        self,
        event_id: str,
        step_key: str,
        step_name: str,
        ok: bool,
        message: str,
        error: Optional[Exception] = None,
        partial: bool = False,
    ):
        status = "success" if ok and not partial else "partial" if partial else "failed"
        if status != "success":
            with self._lock:
                self._degraded = True

        self.store.finish_step(
            task_id=self.task_id,
            event_id=event_id,
            status=status,
            message=message,
            progress=self._end_progress.get(step_key, 90),
            error_type=type(error).__name__ if error else None,
            error_message=str(error) if error else None,
        )

    def final_status(self) -> str:
        with self._lock:
            return "partial" if self._degraded else "succeeded"


task_store = InMemoryTaskStore()
