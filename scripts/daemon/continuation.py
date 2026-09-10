"""Explicit, owner-bound LoRA resume/extension; no GUI preset re-evaluation."""

from __future__ import annotations

import copy
import hashlib
import json
import secrets
import time
from pathlib import Path

import toml

from library.io.output_layout import resolve_output_layout
from library.training.continuation import (
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


class Continuations:
    def __init__(self, manager):
        self.manager = manager
        self.plans: dict[str, dict] = {}

    def latest(self, task_id: str) -> Job:
        with self.manager._lock:
            source = self.manager._jobs.get(task_id)
            if source is None:
                raise ValueError("Training task no longer exists")
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
            raise ValueError("缺少原任务配置快照，不能安全恢复")
        return toml.load(job.config_file)

    @staticmethod
    def run_budget(job: Job, cfg: dict) -> int:
        if job.continuation:
            return int(job.continuation["target_steps"])
        if job.progress_path and Path(job.progress_path).is_file():
            with Path(job.progress_path).open(encoding="utf-8") as fh:
                for index, line in enumerate(fh):
                    if index > 1000:
                        break
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if event.get("ev") == "run_start" and event.get("total_steps"):
                        return int(event["total_steps"])
        steps = cfg.get("max_train_steps") or job.target_steps
        if steps:
            return int(steps)
        raise ValueError("缺少原训练的有效总步数，无法确定恢复预算")

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
            raise ValueError("该任务仍在运行或排队")
        root = job.root_job_id or job.id
        if any(
            (item.root_job_id or item.id) == root and item.state not in TERMINAL_STATES
            for item in self.manager.list_jobs()
        ):
            raise ValueError("该训练已有活动任务")
        selected = None
        attempts = self._attempts(job)
        for index, attempt in enumerate(attempts):
            try:
                cfg = self.snapshot(attempt)
            except ValueError:
                if index == 0 and (attempt.continuation or attempt.recovery_state):
                    raise ValueError("最新一次训练没有完整状态，不能自动回退到旧状态")
                continue
            if (
                self.variant(attempt) != "lora"
                or cfg.get("network_module") != "networks.lora_anima"
            ):
                raise ValueError("首版仅支持普通 LoRA 训练")
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
                raise ValueError("此训练配置暂不支持完整状态加训")
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
                        raise ValueError("分阶段训练暂不支持此入口")
                    owned.append((int(data["global_step"]), path, data, cfg, attempt))
                except ValueError as exc:
                    if str(exc) == "分阶段训练暂不支持此入口":
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
                raise ValueError("最新一次训练没有完整状态，不能自动回退到旧状态")
        if selected is None:
            raise ValueError("缺少完整且归属匹配的训练状态")
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
                    raise ValueError(f"原训练文件不存在：{key}")
        if cfg.get("lr_scheduler_type") or cfg.get("lr_scheduler_args"):
            raise ValueError("自定义学习率调度器暂不支持此入口")
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
            if source.state == STATE_DONE and cfg.get(
                "lr_scheduler", "constant"
            ) not in {
                "constant",
                "constant_with_warmup",
            }:
                raise ValueError("此学习率调度器暂不支持增加训练目标")
        except (ValueError, OSError, KeyError, TypeError) as exc:
            candidate.update(available=False, reason=str(exc))
        return candidate, cfg

    def candidates(self, variant: str) -> dict:
        roots = {}
        for job in self.manager.list_jobs():
            root = job.root_job_id or job.id
            if root not in roots or self.manager._attempt_sort_key(
                job
            ) > self.manager._attempt_sort_key(roots[root]):
                roots[root] = job
        return {
            "candidates": [
                self.describe(job)[0]
                for job in sorted(
                    roots.values(), key=lambda j: j.submitted_at, reverse=True
                )
                if self.variant(job) == variant
            ]
        }

    def detail(self, task_id: str) -> dict:
        candidate, cfg = self.describe(self.latest(task_id))
        return {"candidate": candidate, "snapshot": cfg or {}}

    def prepare(self, task_id: str, body: dict) -> dict:
        latest = self.latest(task_id)
        cfg, state_dir, state, job = self.inputs(latest)
        if job.id != body.get("source_job_id"):
            raise ValueError("任务已有新的训练记录，请刷新选择")
        key = "max_train_steps" if cfg.get("max_train_steps") else "max_train_epochs"
        target = body.get("target")
        old = int(cfg.get(key) or 0)
        if (
            body.get("budget_key") != key
            or type(target) is not int
            or target < old
            or target <= 0
        ):
            raise ValueError("训练目标必须是正整数，且不能小于原目标")
        old_steps = self.run_budget(job, cfg)
        if key == "max_train_epochs" and (old <= 0 or old_steps % old):
            raise ValueError("无法从原训练恢复每轮步数")
        target_steps = (
            target if key == "max_train_steps" else target * (old_steps // old)
        )
        mode = "extend" if target > old else "resume"
        if target_steps <= int(state["global_step"]) or (
            job.state == STATE_DONE and mode != "extend"
        ):
            raise ValueError("训练已达到原目标，请提高最大轮次或步数")
        if mode == "extend" and cfg.get("lr_scheduler", "constant") not in {
            "constant",
            "constant_with_warmup",
        }:
            raise ValueError("此学习率调度器暂不支持增加训练目标")
        warmup = cfg.get("lr_warmup_steps", 0)
        warmup = int(warmup * old_steps) if isinstance(warmup, float) else warmup
        scheduler_steps = old_steps
        signature_output_dir = str(
            resolve_output_layout(cfg.get("output_dir"), cfg.get("output_name")).root
        )
        if job.continuation:
            scheduler_steps = job.continuation["scheduler_steps"]
            warmup = job.continuation["warmup_steps"]
            signature_output_dir = job.continuation["signature_output_dir"]
        blueprint_path = job.dir / "dataset.snapshot.json"
        if blueprint_path.is_file():
            blueprint = json.loads(blueprint_path.read_text())
        else:
            blueprint = (job.continuation or {}).get(
                "dataset_blueprint"
            ) or blueprint_from_config(cfg)
        plan = dict(
            source_job_id=job.id,
            root_job_id=job.root_job_id or job.id,
            config_snapshot=str(Path(job.config_file).resolve()),
            mode=mode,
            budget_key=key,
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

    def submit(self, task_id: str, body: dict) -> Job:
        token = body.get("token")
        if not isinstance(token, str) or body.get("idempotency_key") != token:
            raise ValueError("无效的续训提交标识")
        with self.manager._lock:
            for existing in self.manager._jobs.values():
                if (
                    existing.continuation
                    and existing.continuation.get("token") == token
                ):
                    if (existing.root_job_id or existing.id) != task_id:
                        raise ValueError("续训标识不属于此任务")
                    return existing
            plan = self.plans.get(token)
            if (
                not plan
                or plan["expires"] < time.time()
                or plan["root_job_id"] != task_id
            ):
                raise ValueError("续训准备已过期，请重新点击训练")
            root = plan["root_job_id"]
            if root in self.manager._resuming_roots:
                raise ValueError("该任务正在提交续训")
            latest = self.latest(task_id)
            self.manager._resuming_roots.add(root)
        try:
            cfg, state_dir, state, source = self.inputs(latest)
            if source.id != plan["source_job_id"]:
                raise ValueError("原任务已有新的训练记录")
            if (
                hashlib.sha256(Path(source.config_file).read_bytes()).hexdigest()
                != plan["snapshot_digest"]
                or str(state_dir.resolve()) != plan["state_dir"]
                or state_fingerprint(state_dir) != plan["fingerprint"]
            ):
                raise ValueError("原训练状态或配置已变化，请刷新后重试")
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
                raise ValueError("复制恢复状态时来源发生变化")
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
            child.config_signature = source.config_signature
            child.dataset_signature = source.dataset_signature
            child.target_steps = plan["target_steps"]
            # Register only after all immutable inputs are ready. Preserve the
            # queue's current pause state (start=None).
            with self.manager._lock:
                self.manager._register_and_queue(child, start=None)
            return child
        finally:
            with self.manager._lock:
                self.manager._resuming_roots.discard(root)
