#!/usr/bin/env node

/**
 * Validate the checked-in frontend lockfile and, when requested, repair the
 * local install from that lockfile.  This intentionally never runs lifecycle
 * scripts: esbuild and its platform packages are already pinned in the lock.
 */

import { existsSync, readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const frontendDir = resolve(scriptDir, "../../webui/frontend");
const lockPath = resolve(frontendDir, "package-lock.json");
const packagePath = resolve(frontendDir, "package.json");
const approvedHosts = new Set(
  (process.env.ANIMA_NPM_ALLOWED_HOSTS || "registry.npmmirror.com,registry.npmjs.org")
    .split(",")
    .map((host) => host.trim().toLowerCase())
    .filter(Boolean),
);
const allowedLifecyclePackages = new Set([
  "node_modules/@parcel/watcher",
  "node_modules/esbuild",
  "node_modules/fsevents",
]);
let validatedLock = null;

function containsBlockedPackage(value) {
  // Check path components instead of matching only the unscoped name.  This
  // catches both `keyv` and scoped/aliased forms such as `@keyv/redis`.
  return String(value)
    .replaceAll("\\", "/")
    .split("/")
    .some((component) => component.toLowerCase() === "keyv" || component.toLowerCase() === "@keyv");
}

function hasUnsafeDependencySpec(value) {
  const spec = String(value).trim().toLowerCase();
  return /^(?:npm:|file:|link:|workspace:|git:|git\\+|github:|https?:|ssh:)/.test(spec);
}

function validIntegrity(value) {
  const integrity = String(value || "");
  if (!integrity.startsWith("sha512-")) return false;
  const encoded = integrity.slice("sha512-".length);
  if (!/^[A-Za-z0-9+/]+={0,2}$/.test(encoded)) return false;
  try {
    return Buffer.from(encoded, "base64").length === 64;
  } catch {
    return false;
  }
}

function fail(message) {
  console.error(`[npm-security] ${message}`);
  process.exitCode = 1;
}

function validateLock() {
  let lock;
  let manifest;
  try {
    lock = JSON.parse(readFileSync(lockPath, "utf8"));
    manifest = JSON.parse(readFileSync(packagePath, "utf8"));
  } catch (error) {
    fail(`cannot read ${lockPath}: ${error.message}`);
    return false;
  }
  if (lock.lockfileVersion !== 3) {
    fail(`unsupported npm lockfileVersion=${lock.lockfileVersion}; expected v3`);
    return false;
  }
  const packages = lock.packages && typeof lock.packages === "object" ? lock.packages : {};
  const errors = [];
  const hooks = [];
  const root = packages[""] || {};
  if (root.name !== manifest.name || root.version !== manifest.version) {
    errors.push(`lock root identity does not match package.json (${root.name}@${root.version} vs ${manifest.name}@${manifest.version})`);
  }
  for (const field of ["dependencies", "devDependencies", "optionalDependencies"]) {
    const expected = manifest[field] || {};
    const actual = root[field] || {};
    for (const [name, version] of Object.entries(expected)) {
      if (containsBlockedPackage(name)) {
        errors.push(`blocked package in package.json: ${name}`);
      }
      if (actual[name] !== version) errors.push(`lock root ${field} drift: ${name}`);
      if (hasUnsafeDependencySpec(version)) {
        errors.push(`unsupported dependency spec in package.json: ${name}=${version}`);
      }
    }
    for (const name of Object.keys(actual)) {
      if (!(name in expected)) errors.push(`lock root ${field} has undeclared package: ${name}`);
    }
  }
  for (const [name, info] of Object.entries(packages)) {
    if (name && containsBlockedPackage(name)) {
      errors.push(`blocked package in lockfile: ${name}`);
    }
    if (!info || typeof info !== "object") {
      if (name) errors.push(`invalid lock package entry: ${name}`);
      continue;
    }
    for (const field of ["dependencies", "optionalDependencies", "peerDependencies", "devDependencies"]) {
      for (const [dependencyName, spec] of Object.entries(info[field] || {})) {
        if (containsBlockedPackage(dependencyName)) {
          errors.push(`blocked dependency in lockfile: ${name} -> ${dependencyName}`);
        }
        if (hasUnsafeDependencySpec(spec)) {
          errors.push(`unsupported dependency spec in lockfile: ${name} -> ${dependencyName}=${spec}`);
        }
      }
    }
    if (name && !info.resolved) {
      errors.push(`missing pinned tarball URL: ${name}`);
    }
    if (name && !info.integrity) {
      errors.push(`missing integrity digest: ${name}`);
    }
    if (info.resolved) {
      let url;
      try {
        url = new URL(String(info.resolved));
      } catch {
        url = null;
      }
      if (
        !url ||
        url.protocol !== "https:" ||
        !approvedHosts.has(url.hostname.toLowerCase()) ||
        url.username ||
        url.password ||
        url.port ||
        url.search ||
        url.hash ||
        !url.pathname.endsWith(".tgz")
      ) {
        errors.push(`unapproved tarball URL: ${name} -> ${info.resolved}`);
      }
    }
    if (info.resolved && !validIntegrity(info.integrity)) {
      errors.push(`missing SHA512 integrity: ${name}`);
    }
    if (info.hasInstallScript === true) {
      hooks.push(name || "<root>");
      if (!allowedLifecyclePackages.has(name)) {
        errors.push(`unreviewed lifecycle script in lockfile: ${name}`);
      }
    }
    if (info.name && containsBlockedPackage(info.name)) {
      errors.push(`blocked package metadata in lockfile: ${name} (${info.name})`);
    }
    if (info.resolved && containsBlockedPackage(info.resolved)) {
      errors.push(`blocked package name in tarball URL: ${name}`);
    }
  }
  if (errors.length) {
    for (const error of errors) fail(error);
    return false;
  }
  validatedLock = lock;
  console.log(`[npm-security] lock verified: ${Object.keys(packages).length} packages; lifecycle hooks=${hooks.length}`);
  if (hooks.length) {
    console.log(`[npm-security] hooks remain disabled by --ignore-scripts: ${hooks.join(", ")}`);
  }
  return true;
}

function npmCommand() {
  return process.platform === "win32" ? "npm.cmd" : "npm";
}

function runNpm(args) {
  return spawnSync(npmCommand(), args, {
    cwd: frontendDir,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
}

function treeHasBlockedPackage(value, path = "") {
  if (!value || typeof value !== "object") return false;
  if (path && containsBlockedPackage(path)) return true;
  for (const [name, dependency] of Object.entries(value.dependencies || {})) {
    const childPath = path ? `${path}/${name}` : name;
    if (containsBlockedPackage(name) || treeHasBlockedPackage(dependency, childPath)) return true;
  }
  return false;
}

function installedLockMatches() {
  if (!validatedLock) return false;
  const localLockPath = resolve(frontendDir, "node_modules/.package-lock.json");
  if (!existsSync(localLockPath)) return false;
  let localLock;
  try {
    localLock = JSON.parse(readFileSync(localLockPath, "utf8"));
  } catch {
    return false;
  }
  const expectedPackages = validatedLock.packages || {};
  for (const [path, info] of Object.entries(localLock.packages || {})) {
    if (!path) continue;
    if (containsBlockedPackage(path)) return false;
    const expected = expectedPackages[path];
    if (!expected || expected.version !== info.version) return false;
    if (expected.resolved !== info.resolved || expected.integrity !== info.integrity) return false;
  }
  return true;
}

function treeIsHealthy() {
  const result = runNpm(["ls", "--all", "--ignore-scripts", "--offline", "--json"]);
  let tree = {};
  try {
    tree = JSON.parse(result.stdout || "{}");
  } catch {
    return false;
  }
  if (result.status !== 0 || (Array.isArray(tree.problems) && tree.problems.length)) {
    return false;
  }
  if (treeHasBlockedPackage(tree) || !installedLockMatches()) return false;
  // With lifecycle scripts disabled, esbuild must still resolve its pinned
  // platform package. Otherwise its install hook could fall back to a dynamic
  // network install during the first build.
  const probe = spawnSync(
    process.execPath,
    [
      "-e",
      `const e=require('esbuild');
const r=require('rollup');
if (!e.version || typeof r.rollup !== 'function') process.exit(1);
const transformed=e.transformSync('const x=1',{loader:'js'});
if (!transformed.code || !transformed.code.includes('x')) process.exit(1);
(async()=>{
  const b=await r.rollup({input:'__monadforge_probe__',plugins:[{
    resolveId(id){return id==='__monadforge_probe__'?id:null},
    load(id){return id==='__monadforge_probe__'?'export const value=1;':null}
  }]});
  const out=await b.generate({format:'es'});
  await b.close();
  if (!out.output?.[0]?.code?.includes('value')) process.exit(1);
})().catch(()=>process.exit(1));`,
    ],
    {
      cwd: frontendDir,
      encoding: "utf8",
      stdio: "ignore",
    },
  );
  return probe.status === 0;
}

function installFromLock() {
  const offline = process.argv.includes("--offline") || process.env.ANIMA_NPM_OFFLINE === "1";
  console.log(
    `[npm-security] repairing frontend tree with npm ci --ignore-scripts (${offline ? "offline" : "prefer-offline"} mode)`,
  );
  const result = runNpm([
    "ci",
    "--ignore-scripts",
    offline ? "--offline" : "--prefer-offline",
    "--no-audit",
    "--no-fund",
  ]);
  if (result.stdout) process.stdout.write(result.stdout);
  if (result.stderr) process.stderr.write(result.stderr);
  return result.status === 0;
}

const ensure = process.argv.includes("--ensure");
const force = process.argv.includes("--force") || process.argv.includes("--clean") || process.env.ANIMA_NPM_REINSTALL === "1";
if (!validateLock()) process.exit(1);
if (!ensure) process.exit(0);

if (!force && treeIsHealthy()) {
  console.log("[npm-security] existing frontend tree matches the lockfile");
  process.exit(0);
}
if (!installFromLock() || !treeIsHealthy()) {
  fail("npm ci completed but the installed tree is not a clean lockfile tree");
  process.exit(1);
}
console.log("[npm-security] frontend tree ready");
