/* 官方 Fluent UI Web Components；固定依赖版本，运行时无需联网。 */
import '@fluentui/web-components/button.js';
import { setTheme } from '@fluentui/web-components/theme/set-theme.js';
import { webLightTheme, webDarkTheme } from '@fluentui/tokens';

window.VideoLensFluent = {
  applyTheme(mode) {
    const dark = mode === 'dark';
    const theme = { ...(dark ? webDarkTheme : webLightTheme) };
    // 将 Fluent 默认品牌蓝替换为低饱和中性色，含焦点、悬停、按下态。
    for (const name of Object.keys(theme)) {
      if (/Brand/.test(name) && typeof theme[name] === 'string' && theme[name].startsWith('#')) {
        theme[name] = /Foreground|Stroke/.test(name) ? (dark ? '#c7d3cc' : '#44594d') : (dark ? '#48574e' : '#52665a');
      }
    }
    setTheme(theme);
  },
};
window.VideoLensFluent.applyTheme(document.documentElement.dataset.theme || 'light');
