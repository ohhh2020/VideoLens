/* 一次性探针：验证 Obsidian 插件的纯逻辑（不依赖真实 Obsidian 运行时）。
 * 做法：拦截 require("obsidian") 返回 stub，再把源码尾部追加导出，跑几个断言。
 */
const fs = require("fs");
const path = require("path");
const Module = require("module");
const vm = require("vm");

const ROOT = require("path").resolve(__dirname, "..");
const src = fs.readFileSync(path.join(ROOT, "clients/obsidian_plugin/main.js"), "utf8");

const stubObsidian = {
  Plugin: class {},
  Notice: class { setMessage() {} hide() {} },
  Modal: class {},
  PluginSettingTab: class {},
  Setting: class {},
  MarkdownView: class {},
  requestUrl: async () => ({ status: 200, text: "{}", json: {} }),
};

const sandbox = {
  require: (name) => (name === "obsidian" ? stubObsidian : require(name)),
  module: { exports: {} },
  exports: {},
  console,
  setTimeout,
  navigator: { clipboard: { writeText: async () => {} } },
  __filename: path.join(ROOT, "clients/obsidian_plugin/main.js"),
  __dirname: path.join(ROOT, "clients/obsidian_plugin"),
};
sandbox.globalThis = sandbox;

const patched = src + "\nmodule.exports.__t = { renderAnswer, renderVideoNote, hms, humanBytes, truncate, ZhiYingClient, DEFAULT_SETTINGS };\n";
try {
  vm.runInNewContext(patched, sandbox, { filename: "main.js" });
} catch (error) {
  console.log("LOAD FAILED:", error.message);
  process.exit(1);
}

let pass = 0;
let fail = 0;
function check(name, cond, detail) {
  if (cond) {
    pass += 1;
    console.log("  ok  " + name);
  } else {
    fail += 1;
    console.log("  FAIL " + name + (detail === undefined ? "" : " -> " + JSON.stringify(detail)));
  }
}

const T = sandbox.module.exports.__t;

check("hms 毫秒补零", T.hms(3723456) === "01:02:03.456", T.hms(3723456));
check("hms 零值", T.hms(0) === "00:00:00.000", T.hms(0));
check("hms 负数归零", T.hms(-5) === "00:00:00.000", T.hms(-5));
check("humanBytes 字节", T.humanBytes(512) === "512 B", T.humanBytes(512));
check("humanBytes MiB", T.humanBytes(5 * 1024 * 1024) === "5.0 MiB", T.humanBytes(5 * 1024 * 1024));
check("truncate 截断", T.truncate("一二三四五六", 4) === "一二三四…", T.truncate("一二三四五六", 4));
check("truncate 换行折叠", T.truncate("a\n\nb", 10) === "a b", T.truncate("a\n\nb", 10));

const answer = T.renderAnswer({
  question: "切片长度怎么定",
  answer: "默认 10 到 30 秒。\n第二行。",
  video_id: "vTEST",
  backend: "keywords",
  elapsed_ms: 1234,
  degraded: true,
  refs: [
    { index: 1, clip_id: "vTEST-c0000", time_range: "00:00:00.000 – 00:00:20.100", text: "切片默认 10 秒起" },
    { index: 2, clip_id: "vTEST-c0001", start_ms: 18600, end_ms: 40000, text: "重叠 1.5 秒" },
  ],
});
check("问答含 callout", answer.startsWith("> [!quote] 知影问答"), answer.slice(0, 20));
check("问答含问题", answer.includes("> **问**：切片长度怎么定"));
check("问答多行加引用前缀", answer.includes("> 第二行。"));
check("引用表表头", answer.includes("| # | 时间区间 | clip_id | 原文 |"));
check("引用 1 行", answer.includes("| 1 | 00:00:00.000 – 00:00:20.100 | `vTEST-c0000` | 切片默认 10 秒起 |"));
check("引用 2 用 hms 兜底", answer.includes("| 2 | 00:00:18.600 – 00:00:40.000 | `vTEST-c0001` | 重叠 1.5 秒 |"), answer);
check("元信息含降级", answer.includes("降级"), answer.split("\n").slice(-1)[0]);

const note = T.renderVideoNote(
  { video_id: "vTEST", title: "演示", status: "ready", duration_ms: 40000, width: 1280, height: 720, fps: 25, source_url: "https://x/y" },
  [{ clip_id: "vTEST-c0000", start_ms: 0, end_ms: 20100, core_text: "第一段", keyframe_ids: ["kf1", "kf2"] }],
  [{ keyframe_id: "kf1", timestamp_ms: 5000, path: "D:/f/kf1.jpg", description: "彩条" }],
  { summary: "这是一段测试视频。" }
);
check("笔记含 frontmatter video_id", note.includes("zhiying_video_id: vTEST"));
check("笔记含总结", note.includes("## 内容总结") && note.includes("这是一段测试视频。"));
check("笔记含时间线标题", note.includes("### 00:00:00.000 – 00:00:20.100 · `vTEST-c0000`"));
check("笔记含关键帧列表", note.includes("`kf1` · 00:00:05.000 · D:/f/kf1.jpg"));
check("笔记含关键帧描述", note.includes("  - 彩条"));

// 客户端：错误路径要翻成人话（连不上）
(async () => {
  const client = new T.ZhiYingClient({
    settings: { baseUrl: "http://127.0.0.1:9", token: "", pollIntervalMs: 10, timeoutMs: 1000 },
  });
  check("URL 拼接去尾斜杠", client.url("/api/health") === "http://127.0.0.1:9/api/health", client.url("/api/health"));
  const err = await client.call("GET", "/api/health").then(() => null).catch((e) => e.message);
  check("连不上时给出启动提示", err === null || err.includes("连不上知影服务"), err);

  console.log(`\nPASS=${pass} FAIL=${fail}`);
  process.exit(fail === 0 ? 0 : 1);
})();
