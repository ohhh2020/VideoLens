/*
 * 知影 · Obsidian 插件
 *
 * 为什么不打包：插件只调本地 HTTP 接口，全部逻辑加起来几百行，直接用
 * CommonJS 写 `main.js` 就能被 Obsidian 加载（不需要 esbuild/rollup 工具链）。
 * 别人把整个目录拷进 `.obsidian/plugins/` 就能用 —— 对一个课程作业来说，
 * "能装、能跑、能改" 比"工程化漂亮"重要。
 *
 * 三个关键实现细节
 * ----------------
 * 1. 一律用 Obsidian 的 `requestUrl` 而不是 `fetch`：桌面端 `file://`/`app://`
 *    上下文里 `fetch` 会被 CORS 拦，`requestUrl` 走 Electron 主进程不受限。
 * 2. `Authorization: Bearer <token>`：知影默认只监听回环地址，但如果用户配了
 *    token，请求必须带上，否则 401。
 * 3. 时间戳原样展示 `HH:MM:SS.mmm`：需求文档要求引用误差 ≤2s，写进笔记的
 *    时间必须能点回去核对，不能四舍五入成 "第 3 分钟"。
 */

const { Plugin, Notice, Modal, PluginSettingTab, Setting, requestUrl, MarkdownView } =
  require("obsidian");

const DEFAULT_SETTINGS = {
  baseUrl: "http://127.0.0.1:8000",
  token: "",
  pollIntervalMs: 1200,
  timeoutMs: 45 * 60 * 1000,
  autoInsert: true,
  insertHeading: "## 知影问答",
};

/* ------------------------------------------------------------------------- */
/* 工具                                                                      */
/* ------------------------------------------------------------------------- */

function hms(ms) {
  const total = Math.max(0, Math.floor(Number(ms) || 0));
  const h = Math.floor(total / 3600000);
  const m = Math.floor((total % 3600000) / 60000);
  const s = Math.floor((total % 60000) / 1000);
  const milli = total % 1000;
  const pad = (n, w) => String(n).padStart(w, "0");
  return `${pad(h, 2)}:${pad(m, 2)}:${pad(s, 2)}.${pad(milli, 3)}`;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function humanDuration(ms) {
  const sec = Math.round((Number(ms) || 0) / 1000);
  if (sec < 60) return `${sec} 秒`;
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min} 分 ${sec % 60} 秒`;
  return `${Math.floor(min / 60)} 小时 ${min % 60} 分`;
}

function humanBytes(size) {
  let value = Number(size) || 0;
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  for (let i = 0; i < units.length; i += 1) {
    if (Math.abs(value) < 1024 || i === units.length - 1) {
      return i === 0 ? `${Math.round(value)} B` : `${value.toFixed(1)} ${units[i]}`;
    }
    value /= 1024;
  }
  return `${value.toFixed(1)} TiB`;
}

function truncate(text, max) {
  const value = String(text || "").replace(/\s+/g, " ").trim();
  return value.length > max ? value.slice(0, max) + "…" : value;
}

/* ------------------------------------------------------------------------- */
/* HTTP 客户端                                                               */
/* ------------------------------------------------------------------------- */

class ZhiYingClient {
  constructor(plugin) {
    this.plugin = plugin;
  }

  get headers() {
    const headers = { "Content-Type": "application/json", Accept: "application/json" };
    const token = (this.plugin.settings.token || "").trim();
    if (token) headers.Authorization = `Bearer ${token}`;
    return headers;
  }

  url(path) {
    const base = (this.plugin.settings.baseUrl || "").replace(/\/+$/, "");
    return `${base}${path}`;
  }

  async call(method, path, body) {
    let response;
    try {
      response = await requestUrl({
        url: this.url(path),
        method,
        headers: this.headers,
        body: body === undefined ? undefined : JSON.stringify(body),
        throw: false,
      });
    } catch (error) {
      throw new Error(
        `连不上知影服务（${this.plugin.settings.baseUrl}）。先在终端跑 ` +
          `"python -m clients.cli serve" 或 "uvicorn api.main:app"。原始错误：${error.message}`
      );
    }
    if (response.status >= 400) {
      let detail = "";
      try {
        const payload = response.json;
        detail = typeof payload === "object" ? JSON.stringify(payload).slice(0, 300) : "";
      } catch (error) {
        detail = String(response.text || "").slice(0, 300);
      }
      throw new Error(`${method} ${path} → HTTP ${response.status} ${detail}`);
    }
    if (response.status === 204 || !response.text) return {};
    try {
      return response.json;
    } catch (error) {
      throw new Error(`${method} ${path} 返回的不是 JSON：${String(response.text).slice(0, 200)}`);
    }
  }

  health() {
    return this.call("GET", "/api/health");
  }

  listVideos(limit = 50) {
    return this.call("GET", `/api/videos?limit=${encodeURIComponent(limit)}`);
  }

  video(videoId) {
    return this.call("GET", `/api/videos/${encodeURIComponent(videoId)}`);
  }

  stats() {
    return this.call("GET", "/api/stats");
  }

  analyze(source, options = {}) {
    return this.call("POST", "/api/videos/analyze", { source, ...options });
  }

  task(taskId) {
    return this.call("GET", `/api/tasks/${encodeURIComponent(taskId)}`);
  }

  ask(videoId, question, topK) {
    const body = { question };
    if (topK) body.top_k = Number(topK);
    return this.call("POST", `/api/videos/${encodeURIComponent(videoId)}/ask`, body);
  }

  /** 提交任务并轮询到终态；onProgress(stage, progress, message) 用于刷新 Notice。 */
  async analyzeAndWait(source, options, onProgress) {
    const started = await this.analyze(source, options);
    const taskId = started.task_id;
    if (!taskId) return started; // 服务端同步返回了结果
    const deadline = Date.now() + this.plugin.settings.timeoutMs;
    let lastProgress = -1;
    while (Date.now() < deadline) {
      const task = await this.task(taskId);
      const progress = Number(task.progress || 0);
      if (progress - lastProgress >= 0.02 && onProgress) {
        onProgress(task.stage || "", progress, task.message || "");
      }
      lastProgress = progress;
      if (["done", "failed", "cancelled"].includes(task.status)) {
        if (task.status !== "done") {
          throw new Error(`分析任务 ${task.status}：${task.error || "未知原因"}`);
        }
        return task.report || task;
      }
      await sleep(this.plugin.settings.pollIntervalMs);
    }
    throw new Error("分析超时（可在设置里调大超时时间）");
  }
}

/* ------------------------------------------------------------------------- */
/* 弹窗                                                                      */
/* ------------------------------------------------------------------------- */

class InputModal extends Modal {
  constructor(app, title, label, placeholder, onSubmit) {
    super(app);
    this.titleText = title;
    this.label = label;
    this.placeholder = placeholder;
    this.onSubmit = onSubmit;
    this.value = "";
  }

  onOpen() {
    const { contentEl } = this;
    contentEl.createEl("h3", { text: this.titleText });
    contentEl.createEl("p", { text: this.label, cls: "zhiying-hint" });
    const input = contentEl.createEl("input", {
      type: "text",
      placeholder: this.placeholder,
      cls: "zhiying-input",
    });
    input.addEventListener("input", () => {
      this.value = input.value;
    });
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        this.submit();
      }
    });
    const row = contentEl.createDiv({ cls: "zhiying-row" });
    const ok = row.createEl("button", { text: "确定", cls: "mod-cta" });
    ok.addEventListener("click", () => this.submit());
    const cancel = row.createEl("button", { text: "取消" });
    cancel.addEventListener("click", () => this.close());
    setTimeout(() => input.focus(), 50);
  }

  submit() {
    const value = (this.value || "").trim();
    if (!value) {
      new Notice("知影：内容不能为空");
      return;
    }
    this.close();
    this.onSubmit(value);
  }

  onClose() {
    this.contentEl.empty();
  }
}

class VideoPickerModal extends Modal {
  constructor(app, records, onPick) {
    super(app);
    this.records = records || [];
    this.onPick = onPick;
  }

  onOpen() {
    const { contentEl } = this;
    contentEl.createEl("h3", { text: `知影 · 视频库（${this.records.length}）` });
    if (!this.records.length) {
      contentEl.createEl("p", {
        text: "库里还没有视频。先跑「知影：分析视频」或命令行 analyze。",
      });
      return;
    }
    const list = contentEl.createDiv({ cls: "zhiying-video-list" });
    for (const record of this.records) {
      const item = list.createDiv({ cls: "zhiying-video-item" });
      item.createEl("div", {
        text: record.title || record.display_name || record.video_id,
        cls: "zhiying-video-title",
      });
      item.createEl("div", {
        text: `${record.video_id} · ${record.status} · ${hms(record.duration_ms)}`,
        cls: "zhiying-video-meta",
      });
      item.addEventListener("click", () => {
        this.close();
        this.onPick(record);
      });
    }
  }

  onClose() {
    this.contentEl.empty();
  }
}

/* ------------------------------------------------------------------------- */
/* Markdown 渲染                                                             */
/* ------------------------------------------------------------------------- */

function renderAnswer(result) {
  const lines = [];
  lines.push(`> [!quote] 知影问答`);
  lines.push(`> **问**：${result.question || ""}`);
  lines.push(`> `);
  for (const line of String(result.answer || "").split("\n")) {
    lines.push(`> ${line}`);
  }
  lines.push("");
  const refs = result.refs || [];
  if (refs.length) {
    lines.push("**引用证据**");
    lines.push("");
    lines.push("| # | 时间区间 | clip_id | 原文 |");
    lines.push("| --- | --- | --- | --- |");
    for (const ref of refs) {
      const range = ref.time_range || `${hms(ref.start_ms)} – ${hms(ref.end_ms)}`;
      const text = truncate(ref.text || "", 120).replace(/\|/g, "\\|");
      lines.push(`| ${ref.index} | ${range} | \`${ref.clip_id}\` | ${text} |`);
    }
    lines.push("");
  }
  const meta = [
    `后端 ${result.backend || "-"}`,
    `${Math.round(result.elapsed_ms || 0)}ms`,
    `视频 \`${result.video_id || ""}\``,
  ];
  if (result.degraded) meta.push("降级");
  lines.push(`*${meta.join(" · ")}*`);
  return lines.join("\n");
}

function renderVideoNote(video, clips, keyframes, report) {
  const lines = [];
  const title = video.title || video.display_name || video.video_id;
  lines.push("---");
  lines.push(`zhiying_video_id: ${video.video_id}`);
  lines.push(`zhiying_status: ${video.status}`);
  lines.push(`zhiying_duration_ms: ${video.duration_ms || 0}`);
  lines.push("---");
  lines.push("");
  lines.push(`# ${title}`);
  lines.push("");
  lines.push("| 项 | 值 |");
  lines.push("| --- | --- |");
  lines.push(`| video_id | \`${video.video_id}\` |`);
  lines.push(`| 时长 | ${hms(video.duration_ms)} |`);
  lines.push(`| 分辨率 | ${video.width || 0}×${video.height || 0} @ ${video.fps || 0}fps |`);
  lines.push(`| 切片 / 关键帧 | ${(clips || []).length} / ${(keyframes || []).length} |`);
  if (video.source_url) lines.push(`| 来源 | ${video.source_url} |`);
  lines.push("");

  const summary = (report && report.summary) || video.summary || "";
  if (summary) {
    lines.push("## 内容总结");
    lines.push("");
    lines.push(summary);
    lines.push("");
  }

  if ((clips || []).length) {
    lines.push("## 时间线");
    lines.push("");
    for (const clip of clips) {
      lines.push(`### ${hms(clip.start_ms)} – ${hms(clip.end_ms)} · \`${clip.clip_id}\``);
      lines.push("");
      const body = clip.core_text || clip.text || "";
      if (body) lines.push(body);
      const ids = clip.keyframe_ids || [];
      if (ids.length) {
        lines.push("");
        lines.push(`关键帧：${ids.map((id) => `\`${id}\``).join("、")}`);
      }
      lines.push("");
    }
  }

  if ((keyframes || []).length) {
    lines.push("## 关键帧");
    lines.push("");
    for (const frame of keyframes) {
      lines.push(`- \`${frame.keyframe_id}\` · ${hms(frame.timestamp_ms)} · ${frame.path}`);
      if (frame.description) lines.push(`  - ${truncate(frame.description, 160)}`);
    }
    lines.push("");
  }
  return lines.join("\n");
}

/* ------------------------------------------------------------------------- */
/* 插件                                                                      */
/* ------------------------------------------------------------------------- */

module.exports = class ZhiYingPlugin extends Plugin {
  async onload() {
    await this.loadSettings();
    this.client = new ZhiYingClient(this);

    this.addCommand({
      id: "zhiying-analyze",
      name: "分析视频（本地路径或 URL）",
      callback: () => this.commandAnalyze(),
    });

    this.addCommand({
      id: "zhiying-ask",
      name: "对当前笔记关联的视频提问",
      callback: () => this.commandAsk(),
    });

    this.addCommand({
      id: "zhiying-list",
      name: "浏览视频库",
      callback: () => this.commandList(),
    });

    this.addCommand({
      id: "zhiying-import",
      name: "把视频报告导入为新笔记",
      callback: () => this.commandImport(),
    });

    this.addCommand({
      id: "zhiying-health",
      name: "检查服务连通性",
      callback: () => this.commandHealth(),
    });

    this.addSettingTab(new ZhiYingSettingTab(this.app, this));

    this.statusBarEl = this.addStatusBarItem();
    this.statusBarEl.setText("知影：未连接");
    this.statusBarEl.addClass("zhiying-status");
    this.refreshStatus();
  }

  onunload() {
    if (this.statusBarEl) this.statusBarEl.remove();
  }

  async loadSettings() {
    this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
  }

  async saveSettings() {
    await this.saveData(this.settings);
    this.refreshStatus();
  }

  async refreshStatus() {
    if (!this.statusBarEl) return;
    try {
      const health = await this.client.health();
      const ready = health && (health.orchestrator || health.status === "ok");
      this.statusBarEl.setText(ready ? "知影：已连接" : "知影：服务就绪（管道未装配）");
      this.statusBarEl.removeClass("zhiying-bad");
      if (!ready) this.statusBarEl.addClass("zhiying-warn");
    } catch (error) {
      this.statusBarEl.setText("知影：未连接");
      this.statusBarEl.addClass("zhiying-bad");
    }
  }

  activeVideoId() {
    // 优先读当前笔记 frontmatter 里的 zhiying_video_id
    const view = this.app.workspace.getActiveViewOfType(MarkdownView);
    if (!view) return "";
    const cache = this.app.metadataCache.getFileCache(view.file);
    const frontmatter = (cache && cache.frontmatter) || {};
    return String(frontmatter.zhiying_video_id || "").trim();
  }

  insertIntoActiveNote(markdown) {
    const view = this.app.workspace.getActiveViewOfType(MarkdownView);
    if (!view) {
      new Notice("知影：没有打开的笔记，结果已复制到剪贴板");
      navigator.clipboard.writeText(markdown).catch(() => {});
      return;
    }
    const editor = view.editor;
    const cursor = editor.getCursor();
    const heading = this.settings.insertHeading || "## 知影问答";
    const block = `\n\n${heading}\n\n${markdown}\n`;
    editor.replaceRange(block, cursor);
    new Notice("知影：已插入当前笔记");
  }

  /* -- 命令实现 --------------------------------------------------------- */

  async commandAnalyze() {
    new InputModal(
      this.app,
      "分析视频",
      "支持本地文件路径（如 D:\\videos\\a.mp4）或视频链接",
      "D:\\videos\\demo.mp4",
      async (source) => {
        const notice = new Notice("知影：正在提交分析…", 0);
        try {
          const report = await this.client.analyzeAndWait(source, {}, (stage, progress, message) => {
            notice.setMessage(
              `知影：${stage || "分析中"} ${Math.round(progress * 100)}% ${message || ""}`
            );
          });
          notice.hide();
          const videoId = report.video_id || "";
          new Notice(
            `知影：完成 ${videoId}（切片 ${report.clip_count || 0} / 关键帧 ${
              report.keyframe_count || 0
            }）`,
            8000
          );
          if (this.settings.autoInsert && videoId) {
            const detail = await this.client.video(videoId);
            const markdown = renderVideoNote(
              detail.video || detail,
              detail.clips || [],
              detail.keyframes || [],
              report
            );
            this.insertIntoActiveNote(markdown);
          }
        } catch (error) {
          notice.hide();
          new Notice(`知影：${error.message}`, 12000);
          console.error("[zhiying]", error);
        }
      }
    ).open();
  }

  async commandAsk() {
    const videoId = this.activeVideoId();
    if (!videoId) {
      new Notice("知影：当前笔记没有 zhiying_video_id。先用「浏览视频库」挑一个，或分析后插入。");
      return;
    }
    new InputModal(this.app, `提问 · ${videoId}`, "对这段视频提问", "切片长度是怎么定的？",
      async (question) => {
        const notice = new Notice("知影：检索中…", 0);
        try {
          const result = await this.client.ask(videoId, question);
          notice.hide();
          this.insertIntoActiveNote(renderAnswer(result));
        } catch (error) {
          notice.hide();
          new Notice(`知影：${error.message}`, 12000);
        }
      }).open();
  }

  async commandList() {
    const notice = new Notice("知影：读取视频库…", 0);
    try {
      const payload = await this.client.listVideos(100);
      notice.hide();
      new VideoPickerModal(this.app, payload.items || [], (record) => {
        new InputModal(
          this.app,
          `提问 · ${record.video_id}`,
          "对这段视频提问（也可以先「把视频报告导入为新笔记」）",
          "这段视频讲了什么？",
          async (question) => {
            const busy = new Notice("知影：检索中…", 0);
            try {
              const result = await this.client.ask(record.video_id, question);
              busy.hide();
              this.insertIntoActiveNote(renderAnswer(result));
            } catch (error) {
              busy.hide();
              new Notice(`知影：${error.message}`, 12000);
            }
          }
        ).open();
      }).open();
    } catch (error) {
      notice.hide();
      new Notice(`知影：${error.message}`, 12000);
    }
  }

  async commandImport() {
    new InputModal(this.app, "导入视频报告", "填 video_id", "v1a2b3c4d5e6f", async (videoId) => {
      const notice = new Notice("知影：读取报告…", 0);
      try {
        const detail = await this.client.video(videoId);
        const video = detail.video || detail;
        const markdown = renderVideoNote(
          video,
          detail.clips || [],
          detail.keyframes || [],
          null
        );
        const title = (video.title || video.display_name || videoId)
          .replace(/[\\/:*?"<>|]/g, "-")
          .slice(0, 60);
        const folder = this.app.vault.getRoot().path;
        const path = `${folder}/${title}.md`.replace(/\/+/g, "/");
        let file = this.app.vault.getAbstractFileByPath(path);
        if (file) {
          await this.app.vault.modify(file, markdown);
        } else {
          file = await this.app.vault.create(path, markdown);
        }
        await this.app.workspace.getLeaf(true).openFile(file);
        notice.hide();
        new Notice(`知影：已生成 ${path}`);
      } catch (error) {
        notice.hide();
        new Notice(`知影：${error.message}`, 12000);
      }
    }).open();
  }

  async commandHealth() {
    try {
      const health = await this.client.health();
      const stats = await this.client.stats().catch(() => null);
      const lines = [
        `服务：${this.settings.baseUrl}`,
        `状态：${health.status || "-"}`,
        `编排器：${health.orchestrator ? "已就绪" : "未装配"}`,
        `鉴权：${health.auth || "-"}`,
      ];
      if (stats) {
        lines.push(
          `库内：视频 ${stats.videos || 0} / 切片 ${stats.clips || 0} / 关键帧 ${
            stats.keyframes || 0
          }`
        );
        lines.push(`关键帧占用：${humanBytes(stats.keyframe_bytes || 0)}`);
        lines.push(`数据库：${humanBytes(stats.db_bytes || 0)}`);
      }
      new Notice("知影 · " + lines.join("\n"), 12000);
      this.refreshStatus();
    } catch (error) {
      new Notice(`知影：${error.message}`, 12000);
    }
  }
};

/* ------------------------------------------------------------------------- */
/* 设置页                                                                    */
/* ------------------------------------------------------------------------- */

class ZhiYingSettingTab extends PluginSettingTab {
  constructor(app, plugin) {
    super(app, plugin);
    this.plugin = plugin;
  }

  display() {
    const { containerEl } = this;
    containerEl.empty();
    containerEl.createEl("h2", { text: "知影 · 设置" });

    new Setting(containerEl)
      .setName("服务地址")
      .setDesc("默认 http://127.0.0.1:8000（只监听本机，数据不出内网）")
      .addText((text) =>
        text
          .setPlaceholder("http://127.0.0.1:8000")
          .setValue(this.plugin.settings.baseUrl)
          .onChange(async (value) => {
            this.plugin.settings.baseUrl = value.trim() || DEFAULT_SETTINGS.baseUrl;
            await this.plugin.saveSettings();
          })
      );

    new Setting(containerEl)
      .setName("API Token")
      .setDesc("服务端配置了 api.token 时必填；留空表示无需鉴权")
      .addText((text) => {
        text.inputEl.type = "password";
        text
          .setPlaceholder("（可留空）")
          .setValue(this.plugin.settings.token)
          .onChange(async (value) => {
            this.plugin.settings.token = value.trim();
            await this.plugin.saveSettings();
          });
      });

    new Setting(containerEl)
      .setName("轮询间隔（毫秒）")
      .setDesc("分析任务进度的刷新频率；太小会增加服务端压力")
      .addText((text) =>
        text
          .setPlaceholder("1200")
          .setValue(String(this.plugin.settings.pollIntervalMs))
          .onChange(async (value) => {
            const parsed = parseInt(value, 10);
            if (!Number.isNaN(parsed) && parsed >= 300) {
              this.plugin.settings.pollIntervalMs = parsed;
              await this.plugin.saveSettings();
            }
          })
      );

    new Setting(containerEl)
      .setName("分析超时（分钟）")
      .setDesc("1 小时视频在 CPU 上可能要 30 分钟，别设太小")
      .addText((text) =>
        text
          .setPlaceholder("45")
          .setValue(String(Math.round(this.plugin.settings.timeoutMs / 60000)))
          .onChange(async (value) => {
            const parsed = parseInt(value, 10);
            if (!Number.isNaN(parsed) && parsed >= 1) {
              this.plugin.settings.timeoutMs = parsed * 60000;
              await this.plugin.saveSettings();
            }
          })
      );

    new Setting(containerEl)
      .setName("分析完成后自动插入笔记")
      .setDesc("把视频摘要与时间线写进当前笔记（带 zhiying_video_id，方便后续提问）")
      .addToggle((toggle) =>
        toggle.setValue(this.plugin.settings.autoInsert).onChange(async (value) => {
          this.plugin.settings.autoInsert = value;
          await this.plugin.saveSettings();
        })
      );

    new Setting(containerEl)
      .setName("插入标题")
      .setDesc("插入问答时的二级标题")
      .addText((text) =>
        text
          .setPlaceholder("## 知影问答")
          .setValue(this.plugin.settings.insertHeading)
          .onChange(async (value) => {
            this.plugin.settings.insertHeading = value || DEFAULT_SETTINGS.insertHeading;
            await this.plugin.saveSettings();
          })
      );

    const row = containerEl.createDiv({ cls: "zhiying-row" });
    const probe = row.createEl("button", { text: "测试连接" });
    probe.addEventListener("click", () => this.plugin.commandHealth());

    containerEl.createEl("p", {
      text:
        "启动服务：python -m clients.cli serve（或 uvicorn api.main:app --port 8000）。" +
        "本插件只与 127.0.0.1 通信，不上传任何视频或转录文本。",
      cls: "zhiying-hint",
    });
  }
}
