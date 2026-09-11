"""Explicit, owner-bound LoRA resume/extension; no GUI preset re-evaluation."""

from __future__ import annotations

import copy
import hashlib
import json
import secrets
import shutil
import time
from pathlib import Path

import toml

from library.io.output_layout import resolve_output_layout
from library.training.continuation import (
    CONSTANT_CONTINUATION_SCHEDULERS,
    CRITICAL_CONTINUATION_PARAMS,
    blueprint_from_config,
    copy_state,
    state_fingerprint,
)
from . import config
from .jobs import (
    Job,
    STATE_DONE,
    STATE_ERROR,
    STATE_STOPPED,
    TERMINAL_STATES,
    new_job_id,
)


class ContinuationError(ValueError):
    """User-facing failure carrying an i18n key for the WebUI to localize."""

    def __init__(self, message: str, *, key: str, **params):
        super().__init__(message)
        self.key = key
        self.params = params


class Continuations:
    #: Candidates listing fingerprints every root's latest state; short-lived
    #: cache keeps the config page responsive without going stale after submit.
    CANDIDATES_TTL = 30.0

    def __init__(self, manager):
        self.manager = manager
        self.plans: dict[str, dict] = {}
        self._candidates_cache: dict[str, tuple[float, dict]] = {}

    def latest(self, task_id: str) -> Job:
        with self.manager._lock:
            source = self.manager._jobs.get(task_id)
            if source is None:
                raise ContinuationError("训练任务不存在", key="task_missing")
            root = source.root_job_id or source.id
            attempts = [
                j
                for j in self.manager._jobs.values()
                if (j.root_job_id or j.id) == root
            ]
            return max(attempts, key=self.manager._attempt_sort_key)

    @staticmethod
    def variant(job: Job) -> str:
        if job.kind == "train":
            return job.method
        if len(job.argv) >= 3 and job.argv[1] == "lora-gui":
            return job.argv[2]
        return ""

    @staticmethod
    def snapshot(job: Job) -> dict:
        if not job.config_file or not Path(job.config_file).is_file():
            raise ContinuationError(
                "缺少原任务配置快照，不能安全恢复", key="missing_snapshot"
            )
        return toml.load(job.config_file)

    @staticmethod
    def run_budget(job: Job, cfg: dict) -> int:
        if job.continuation:
            return int(job.continuation["target_steps"])
        if job.progress_path and Path(job.progress_path).is_file():
            total = None
            with Path(job.progress_path).open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    # The last run_start wins: an earlier interrupted launch in
                    # the same log must not pin the budget of a later rerun.
                    if event.get("ev") == "run_start" and event.get("total_steps"):
                        total = int(event["total_steps"])
            if total:
                return total
        steps = cfg.get("max_train_steps") or job.target_steps
        if steps:
            return int(steps)
        raise ContinuationError(
            "缺少原训练的有效总步数，无法确定恢复预算", key="missing_budget"
        )

    def _attempts(self, job: Job) -> list[Job]:
        root = job.root_job_id or job.id
        return sorted(
            (
                item
                for item in self.manager.list_jobs()
                if (item.root_job_id or item.id) == root
            ),
            key=self.manager._attempt_sort_key,
            reverse=True,
        )

    def _state_dirs(self, cfg: dict) -> list[Path]:
        layout = resolve_output_layout(
            cfg.get("output_dir"), cfg.get("output_name"), cwd=config.ROOT
        )
        names = {layout.name, Path(str(cfg.get("output_name") or layout.name)).name}
        roots = [layout.root, layout.base, layout.base.parent / layout.name]
        continuations = layout.root / "continuations"
        if continuations.is_dir():
            for attempt_dir in continuations.iterdir():
                roots.append(attempt_dir)
                roots.append(attempt_dir / layout.name)
        found: set[Path] = set()
        for root in roots:
            if not root.is_dir():
                continue
            for name in names:
                found.update(root.glob(f"{name}*-state"))
        return sorted(found)

    @staticmethod
    def _read_state(path: Path) -> dict | None:
        try:
            return json.loads((path / "train_state.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def inputs(self, job: Job) -> tuple[dict, Path, dict, Job]:
        if job.state not in {STATE_DONE, STATE_ERROR, STATE_STOPPED}:
            raise ContinuationError("该任务仍在运行或排队", key="task_active")
        root = job.root_job_id or job.id
        if any(
            (item.root_job_id or item.id) == root and item.state not in TERMINAL_STATES
            for item in self.manager.list_jobs()
        ):
            raise ContinuationError("该训练已有活动任务", key="root_active")
        selected = None
        attempts = self._attempts(job)
        for index, attempt in enumerate(attempts):
            try:
                cfg = self.snapshot(attempt)
            except ValueError:
                if index == 0 and (attempt.continuation or attempt.recovery_state):
                    raise ContinuationError(
                        "最新一次训练没有完整状态，不能自动回退到旧状态",
                        key="latest_incomplete",
                    )
                continue
            if (
                self.variant(attempt) != "lora"
                or cfg.get("network_module") != "networks.lora_anima"
            ):
                raise ContinuationError("首版仅支持普通 LoRA 训练", key="lora_only")
            if any(
                cfg.get(key)
                for key in (
                    "use_timestep_mask",
                    "use_moe_style",
                    "use_ortho",
                    "use_lokr",
                    "use_loha",
                    "stage_schedule",
                    "staged_resolution",
                    "train_llm_adapter",
                )
            ):
                raise ContinuationError(
                    "此训练配置暂不支持完整状态加训", key="unsupported_config"
                )
            dirs = set(self._state_dirs(cfg))
            if attempt.continuation:
                dirs.update(
                    self._state_dirs(
                        dict(cfg, output_dir=attempt.continuation["output_dir"])
                    )
                )
            if attempt.recovery_state:
                recovery = Path(attempt.recovery_state)
                if recovery.is_dir():
                    dirs.add(recovery)
            owned = []
            published_incomplete = False
            for path in dirs:
                sidecar = path / "train_state.json"
                if not sidecar.is_file():
                    continue
                data = self._read_state(path)
                if data is None:
                    if self._dir_owned_by(attempt, path):
                        published_incomplete = True
                    continue
                if data.get("job_id") and data.get("job_id") != attempt.id:
                    continue
                if (
                    not data.get("job_id")
                    or data.get("root_job_id") != root
                    or data.get("schema_version", 1) < 3
                ):
                    if self._dir_owned_by(attempt, path):
                        published_incomplete = True
                    continue
                try:
                    if (
                        not attempt.config_signature
                        or not attempt.dataset_signature
                        or data.get("config_signature") != attempt.config_signature
                        or data.get("dataset_signature") != attempt.dataset_signature
                    ):
                        raise ValueError("signature")
                    state_fingerprint(path)
                    if int(data.get("stage_index", -1)) >= 0:
                        raise ContinuationError(
                            "分阶段训练暂不支持此入口", key="staged_unsupported"
                        )
                    owned.append((int(data["global_step"]), path, data, cfg, attempt))
                except ValueError as exc:
                    if "分阶段训练" in str(exc):
                        raise
                    published_incomplete = True
                except (OSError, KeyError, TypeError):
                    published_incomplete = True
            if owned:
                selected = max(
                    owned, key=lambda row: (row[0], row[1].stat().st_mtime_ns)
                )
                break
            if published_incomplete:
                raise ContinuationError(
                    "最新一次训练没有完整状态，不能自动回退到旧状态",
                    key="latest_incomplete",
                )
        if selected is None:
            raise ContinuationError(
                "缺少完整且归属匹配的训练状态", key="no_state"
            )
        _, path, data, cfg, source = selected
        for key in (
            "pretrained_model_name_or_path",
            "qwen3",
            "vae",
            "preprocess_run",
            "dataset_config",
        ):
            if cfg.get(key):
                target = Path(cfg[key])
                if not target.is_absolute():
                    target = config.ROOT / target
                if not target.is_file():
                    raise ContinuationError(
                        f"原训练文件不存在：{key}", key="file_missing", field=key
                    )
        scheduler = self._scheduler_values(source, cfg)
        if scheduler["lr_scheduler_type"] or scheduler["lr_scheduler_args"]:
            raise ContinuationError(
                "自定义学习率调度器暂不支持此入口", key="custom_scheduler"
            )
        self.run_budget(source, cfg)
        return cfg, path, data, source

    @staticmethod
    def _dir_owned_by(attempt: Job, path: Path) -> bool:
        resolved = path.resolve()
        if attempt.recovery_state:
            try:
                if Path(attempt.recovery_state).resolve() == resolved:
                    return True
            except OSError:
                pass
        if attempt.continuation:
            try:
                resolved.relative_to(Path(attempt.continuation["output_dir"]).resolve())
                return True
            except (ValueError, OSError):
                pass
        return False

    @staticmethod
    def _with_resume(argv: list[str], state: Path) -> list[str]:
        out = []
        skip = False
        for token in argv:
            if skip:
                skip = False
                continue
            if token == "--resume":
                skip = True
                continue
            out.append(token)
        out += ["--resume", str(state)]
        return out

    _SCHEDULER_ARG_FIELDS = {
        "--lr_scheduler": ("lr_scheduler", str),
        "--lr_warmup_steps": ("lr_warmup_steps", "_warmup"),
        "--lr_scheduler_type": ("lr_scheduler_type", str),
        "--lr_scheduler_args": ("lr_scheduler_args", str),
    }

    @staticmethod
    def _coerce_scheduler_arg(field: str, raw: str):
        if field == "lr_warmup_steps":
            # Mirror cli_args' int_or_float: >=1 is an absolute step count and
            # must stay an int; a bare float() would silently turn it into a
            # ratio and scale it by the source run's total steps.
            value = float(raw)
            return int(value) if value >= 1 else value
        return raw

    def _effective_snapshot(self, cfg: dict, job: Job) -> dict | None:
        """The trainer-written merged snapshot: the full effective config of the
        source run, CLI overrides included. Falls back to ``None`` when absent
        (e.g. a run that never wrote one), leaving the caller on the sparse
        job-dir snapshot."""
        layouts = [
            resolve_output_layout(
                cfg.get("output_dir"), cfg.get("output_name"), cwd=config.ROOT
            )
        ]
        if job.continuation and job.continuation.get("output_dir"):
            layouts.append(
                resolve_output_layout(
                    job.continuation["output_dir"],
                    cfg.get("output_name"),
                    cwd=config.ROOT,
                )
            )
        for layout in layouts:
            names = {layout.name, Path(str(cfg.get("output_name") or layout.name)).name}
            for name in names:
                path = layout.root / f"{name}.snapshot.toml"
                if path.is_file():
                    try:
                        return toml.load(path)
                    except (OSError, ValueError, TypeError):
                        continue
        return None

    def _scheduler_values(self, job: Job, cfg: dict) -> dict:
        """Authoritative scheduler identity for a source attempt.

        The job-dir config snapshot is the GUI's sparse view: a run submitted
        via daemon HTTP (CLI/MCP) may carry ``--lr_scheduler`` argv overrides
        it never reflects, and preset/method-chain defaults may not appear in
        it at all. Prefer the trainer-written effective snapshot, then apply
        argv flags on top so a CLI override can never slip past the extend
        gate."""
        base = self._effective_snapshot(cfg, job) or cfg
        values = dict(
            lr_scheduler=base.get("lr_scheduler", "constant"),
            lr_warmup_steps=base.get("lr_warmup_steps", 0),
            lr_scheduler_type=base.get("lr_scheduler_type"),
            lr_scheduler_args=base.get("lr_scheduler_args"),
        )
        tokens = list(job.argv)
        index = 0
        while index < len(tokens):
            flag, sep, inline = tokens[index].partition("=")
            if flag not in self._SCHEDULER_ARG_FIELDS:
                index += 1
                continue
            field, _ = self._SCHEDULER_ARG_FIELDS[flag]
            raw = (
                inline if sep else (tokens[index + 1] if index + 1 < len(tokens) else None)
            )
            if raw is not None:
                try:
                    values[field] = self._coerce_scheduler_arg(field, raw)
                except ValueError:
                    values[field] = raw
            index += 1 if sep else 2
        return values

    def describe(self, job: Job) -> tuple[dict, dict | None]:
        candidate = dict(
            task_id=job.root_job_id or job.id,
            job_id=job.id,
            name=job.method,
            variant=self.variant(job),
            preset=job.preset or job.extra_env.get("PRESET", "default"),
            state=job.state,
            submitted_at=job.submitted_at,
            step=0,
            epoch=0,
            available=False,
            reason=None,
            reason_key=None,
            reason_params=None,
            budget_key="max_train_epochs",
            target=0,
        )
        cfg = None
        try:
            cfg = self.snapshot(job)
            candidate["name"] = cfg.get("output_name") or job.method
            key = (
                "max_train_steps" if cfg.get("max_train_steps") else "max_train_epochs"
            )
            candidate.update(budget_key=key, target=int(cfg.get(key) or 0))
            cfg, _, state, source = self.inputs(job)
            key = (
                "max_train_steps" if cfg.get("max_train_steps") else "max_train_epochs"
            )
            candidate.update(
                job_id=source.id,
                state=source.state,
                submitted_at=source.submitted_at,
                name=cfg.get("output_name") or source.method,
                preset=source.preset
                or source.extra_env.get("PRESET", candidate["preset"]),
                variant=self.variant(source) or candidate["variant"],
                budget_key=key,
                target=int(cfg.get(key) or 0),
                step=int(state["global_step"]),
                epoch=int(state.get("current_epoch", 0)),
                available=True,
            )
            if source.state == STATE_DONE and self._scheduler_values(
                source, cfg
            )["lr_scheduler"] not in CONSTANT_CONTINUATION_SCHEDULERS:
                raise ContinuationError(
                    "此学习率调度器暂不支持增加训练目标",
                    key="scheduler_not_extensible",
                )
        except (ValueError, OSError, KeyError, TypeError) as exc:
            candidate.update(
                available=False,
                reason=str(exc),
                reason_key=getattr(exc, "key", None),
                reason_params=getattr(exc, "params", None) or None,
            )
        return candidate, cfg

    def invalidate(self) -> None:
        self._candidates_cache.clear()

    def candidates(self, variant: str) -> dict:
        cached = self._candidates_cache.get(variant)
        if cached and cached[0] > time.time():
            return cached[1]
        roots = {}
        for job in self.manager.list_jobs():
            root = job.root_job_id or job.id
            if root not in roots or self.manager._attempt_sort_key(
                job
            ) > self.manager._attempt_sort_key(roots[root]):
                roots[root] = job
        result = {
            "candidates": [
                self.describe(job)[0]
                for job in sorted(
                    roots.values(), key=lambda j: j.submitted_at, reverse=True
                )
                if self.variant(job) == variant
            ]
        }
        self._candidates_cache[variant] = (
            time.time() + self.CANDIDATES_TTL,
            result,
        )
        return result

    def detail(self, task_id: str) -> dict:
        candidate, cfg = self.describe(self.latest(task_id))
        return {"candidate": candidate, "snapshot": cfg or {}}

    def prepare(self, task_id: str, body: dict) -> dict:
        latest = self.latest(task_id)
        cfg, state_dir, state, job = self.inputs(latest)
        if job.id != body.get("source_job_id"):
            raise ContinuationError(
                "任务已有新的训练记录，请刷新选择", key="stale_attempt"
            )
        key = "max_train_steps" if cfg.get("max_train_steps") else "max_train_epochs"
        target = body.get("target")
        old = int(cfg.get(key) or 0)
        if (
            body.get("budget_key") != key
            or type(target) is not int
            or target < old
            or target <= 0
        ):
            raise ContinuationError(
                "训练目标必须是正整数，且不能小于原目标", key="invalid_target"
            )
        old_steps = self.run_budget(job, cfg)
        if key == "max_train_epochs" and (old <= 0 or old_steps % old):
            raise ContinuationError(
                "无法从原训练恢复每轮步数", key="indivisible_steps"
            )
        target_steps = (
            target if key == "max_train_steps" else target * (old_steps // old)
        )
        mode = "extend" if target > old else "resume"
        if target_steps <= int(state["global_step"]) or (
            job.state == STATE_DONE and mode != "extend"
        ):
            raise ContinuationError(
                "训练已达到原目标，请提高最大轮次或步数", key="target_reached"
            )
        scheduler = self._scheduler_values(job, cfg)
        if mode == "extend" and scheduler["lr_scheduler"] not in (
            CONSTANT_CONTINUATION_SCHEDULERS
        ):
            raise ContinuationError(
                "此学习率调度器暂不支持增加训练目标",
                key="scheduler_not_extensible",
            )
        warmup = scheduler["lr_warmup_steps"]
        warmup = int(warmup * old_steps) if isinstance(warmup, float) else warmup
        scheduler_steps = old_steps
        lr_scheduler = scheduler["lr_scheduler"]
        lr_warmup_steps = scheduler["lr_warmup_steps"]
        signature_output_dir = str(
            resolve_output_layout(cfg.get("output_dir"), cfg.get("output_name")).root
        )
        critical_params = None
        if job.continuation:
            scheduler_steps = job.continuation["scheduler_steps"]
            warmup = job.continuation["warmup_steps"]
            signature_output_dir = job.continuation["signature_output_dir"]
            # A chained extension keeps the identity of the original run.
            lr_scheduler = (
                job.continuation.get("lr_scheduler") or scheduler["lr_scheduler"]
            )
            if job.continuation.get("lr_warmup_steps") is not None:
                lr_warmup_steps = job.continuation["lr_warmup_steps"]
            critical_params = job.continuation.get("critical_params")
        if critical_params is None:
            effective = self._effective_snapshot(cfg, job)
            if effective:
                critical_params = {
                    name: effective[name]
                    for name in CRITICAL_CONTINUATION_PARAMS
                    if name in effective
                }
        blueprint_path = job.dir / "dataset.snapshot.json"
        if blueprint_path.is_file():
            try:
                blueprint = json.loads(blueprint_path.read_text())
            except ValueError:
                raise ContinuationError(
                    "原训练数据蓝图损坏，无法安全恢复", key="blueprint_corrupt"
                )
        else:
            blueprint = (job.continuation or {}).get(
                "dataset_blueprint"
            ) or blueprint_from_config(cfg)
        plan = dict(
            source_job_id=job.id,
            root_job_id=job.root_job_id or job.id,
            config_snapshot=str(Path(job.config_file).resolve()),
            mode=mode,
            continuation_kind=mode,
            budget_key=key,
            source_target=old,
            source_target_steps=old_steps,
            target=target,
            target_steps=target_steps,
            dataset_blueprint=blueprint,
            state_dir=str(state_dir.resolve()),
            fingerprint=state_fingerprint(state_dir),
            snapshot_digest=hashlib.sha256(
                Path(job.config_file).read_bytes()
            ).hexdigest(),
            scheduler_steps=scheduler_steps,
            warmup_steps=warmup,
            lr_scheduler=lr_scheduler,
            lr_warmup_steps=lr_warmup_steps,
            critical_params=critical_params,
            signature_output_dir=signature_output_dir,
            expires=time.time() + 600,
        )
        token = secrets.token_urlsafe(24)
        with self.manager._lock:
            self.plans = {
                k: v for k, v in self.plans.items() if v["expires"] > time.time()
            }
            self.plans[token] = plan
        return {
            "token": token,
            "mode": mode,
            "target_steps": target_steps,
            "recovery_step": state["global_step"],
        }

    def _registered_attempt(self, token: str, task_id: str) -> Job | None:
        """The attempt a token already produced, if any — this survives a
        daemon restart (job.json persists) even though ``plans`` does not."""
        with self.manager._lock:
            for existing in self.manager._jobs.values():
                if (
                    existing.continuation
                    and existing.continuation.get("token") == token
                ):
                    if (existing.root_job_id or existing.id) != task_id:
                        raise ContinuationError(
                            "续训标识不属于此任务", key="token_mismatch"
                        )
                    return existing
        return None

    def _await_root_owner(
        self, root: str, token: str, task_id: str, *, deadline: float
    ) -> Job | None:
        """Take the root reservation, or wait out a concurrent submit carrying
        the same token and return its attempt — idempotency then also covers
        the copy window, not just completed submissions."""
        while True:
            with self.manager._lock:
                existing = self._registered_attempt(token, task_id)
                if existing is not None:
                    return existing
                if root not in self.manager._resuming_roots:
                    self.manager._resuming_roots.add(root)
                    return None
            if time.time() >= deadline:
                raise ContinuationError(
                    "该任务正在提交续训", key="submit_in_progress"
                )
            time.sleep(0.2)

    def submit(self, task_id: str, body: dict) -> Job:
        token = body.get("token")
        if not isinstance(token, str) or body.get("idempotency_key") != token:
            raise ContinuationError("无效的续训提交标识", key="invalid_token")
        existing = self._registered_attempt(token, task_id)
        if existing is not None:
            return existing
        with self.manager._lock:
            plan = self.plans.get(token)
            if (
                not plan
                or plan["expires"] < time.time()
                or plan["root_job_id"] != task_id
            ):
                raise ContinuationError(
                    "续训准备已过期，请重新点击训练", key="token_expired"
                )
            root = plan["root_job_id"]
        existing = self._await_root_owner(root, token, task_id, deadline=plan["expires"])
        if existing is not None:
            return existing
        latest = self.latest(task_id)
        child_id: str | None = None
        try:
            cfg, state_dir, state, source = self.inputs(latest)
            if source.id != plan["source_job_id"]:
                raise ContinuationError(
                    "原任务已有新的训练记录", key="stale_attempt"
                )
            if (
                hashlib.sha256(Path(source.config_file).read_bytes()).hexdigest()
                != plan["snapshot_digest"]
                or str(state_dir.resolve()) != plan["state_dir"]
                or state_fingerprint(state_dir) != plan["fingerprint"]
            ):
                raise ContinuationError(
                    "原训练状态或配置已变化，请刷新后重试", key="state_changed"
                )
            child_id = new_job_id()
            original_layout = resolve_output_layout(
                cfg.get("output_dir"), cfg.get("output_name"), cwd=config.ROOT
            )
            output_dir = original_layout.root / "continuations" / child_id
            private_state = config.job_dir(child_id) / "resume-input"
            private_state.parent.mkdir(parents=True, exist_ok=False)
            digest = state_fingerprint(state_dir, content=True)
            copy_state(state_dir, private_state)
            if (
                state_fingerprint(private_state, content=True) != digest
                or state_fingerprint(state_dir) != plan["fingerprint"]
            ):
                raise ContinuationError(
                    "复制恢复状态时来源发生变化", key="copy_race"
                )
            context = dict(
                plan,
                token=token,
                job_id=child_id,
                state_dir=str(private_state),
                state_digest=digest,
                output_dir=str(output_dir),
                config_signature=state["config_signature"],
                dataset_signature=state["dataset_signature"],
            )
            context_path = config.job_dir(child_id) / "continuation.json"
            context_path.write_text(json.dumps(context, indent=2), encoding="utf-8")
            child_cfg = copy.deepcopy(cfg)
            child_cfg[key := plan["budget_key"]] = plan["target"]
            # Effective step values must not override an epoch-based target.
            if key == "max_train_epochs":
                child_cfg.pop("max_train_steps", None)
            extra_env = {
                k: v
                for k, v in source.extra_env.items()
                if k
                not in {
                    "ANIMA_TRAIN_FRESH",
                    "ANIMA_CONTINUATION_FILE",
                    "GUI_PRESETS",
                    "ARTIST",
                    "PROFILE_STEPS",
                }
            }
            extra_env["ANIMA_CONTINUATION_FILE"] = str(context_path)
            if source.config_file:
                extra_env.setdefault("CONFIG_FILE", source.config_file)
            argv = self._with_resume(list(source.argv), private_state)
            if source.data_manifest and "--preprocess_run" not in argv:
                argv += ["--preprocess_run", source.data_manifest]
            child = Job(
                id=child_id,
                kind="command",
                method="lora-gui",
                preset=source.preset,
                argv=argv,
                extra_env=extra_env,
                root_job_id=root,
                parent_job_id=latest.id,
                attempt_index=latest.attempt_index + 1,
                config_signature=source.config_signature,
                dataset_signature=source.dataset_signature,
                target_steps=plan["target_steps"],
                target_epochs=plan["target"] if key == "max_train_epochs" else None,
                recovery_state=str(private_state),
                recovery_step=state["global_step"],
                continuation=context,
            )
            self.manager._attach_config_file(
                child, config_snapshot=child_cfg, config_file=None
            )
            # _attach_config_file re-derives these from the child snapshot
            # (which keeps the source's legacy epochs for a steps budget);
            # the attempt's contract is the plan, so re-assert it.
            child.config_signature = source.config_signature
            child.dataset_signature = source.dataset_signature
            child.target_steps = plan["target_steps"]
            child.target_epochs = plan["target"] if key == "max_train_epochs" else None
            # Register only after all immutable inputs are ready. Preserve the
            # queue's current pause state (start=None).
            with self.manager._lock:
                self.manager._register_and_queue(child, start=None)
            self.invalidate()
            return child
        except Exception:
            # Never leave a half-written, unregistered job directory behind.
            if child_id:
                with self.manager._lock:
                    registered = child_id in self.manager._jobs
                if not registered:
                    shutil.rmtree(config.job_dir(child_id), ignore_errors=True)
            raise
        finally:
            with self.manager._lock:
                self.manager._resuming_roots.discard(root)
