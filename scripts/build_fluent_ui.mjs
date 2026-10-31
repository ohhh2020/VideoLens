/* 开发时运行；构建依赖置于独立运行目录，不进入源码/Conda 环境。 */
import { createRequire } from 'node:module';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const prefix = resolve(process.argv[2] || resolve(root, '../VideoLens-runtime/cache/ui-build'));
const require = createRequire(resolve(prefix, 'package.json'));
const { build } = require('esbuild');
await build({
  entryPoints: [resolve(root, 'frontend/workbench/fluent-entry.js')],
  outfile: resolve(root, 'frontend/vendor/fluent-ui.min.js'),
  nodePaths: [resolve(prefix, 'node_modules')], bundle: true, minify: true,
  format: 'iife', target: ['es2022'], legalComments: 'inline',
});
const versionsFile = resolve(root, 'frontend/vendor/versions.json');
const versions = JSON.parse(await readFile(versionsFile, 'utf8'));
versions['fluent-ui'] = {
  version: '3.1.3', tokens: '1.0.0-alpha.24',
  source: 'https://github.com/microsoft/fluentui/tree/master/packages/web-components',
  sha256: createHash('sha256').update(await readFile(resolve(root, 'frontend/vendor/fluent-ui.min.js'))).digest('hex'),
};
await writeFile(versionsFile, JSON.stringify(versions, null, 2) + '\n');
console.log('已构建本地 Fluent UI 组件。');
