(function () {
  'use strict';
  var doc = document;
  var root = doc.documentElement;
  var ROOT = doc.body.getAttribute('data-root') || '';
  var SITE = 'https://goofymoocow.github.io/jax';
  var views = Array.prototype.slice.call(doc.querySelectorAll('.view'));
  var SPA = views.length > 0;

  function $all(sel, ctx) { return Array.prototype.slice.call((ctx || doc).querySelectorAll(sel)); }
  function noop() {}
  function esc(s) { return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
  function keyOf(route) { return route ? route.replace(/\//g, '-') : 'home'; }

  // ---------- theme ----------
  function currentTheme() {
    var a = root.getAttribute('data-theme');
    if (a) return a;
    return window.matchMedia && matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  $all('.theme-toggle').forEach(function (b) {
    b.addEventListener('click', function () {
      var next = currentTheme() === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem('jax-theme', next); } catch (e) {}
    });
  });

  // ---------- mobile menu ----------
  var top = doc.querySelector('.top');
  var menuBtn = doc.querySelector('.menu-btn');
  function setMenu(open) {
    if (!top || !menuBtn) return;
    top.classList.toggle('open', open);
    menuBtn.setAttribute('aria-expanded', open ? 'true' : 'false');
  }
  if (top && menuBtn) {
    menuBtn.addEventListener('click', function () { setMenu(!top.classList.contains('open')); });
    doc.addEventListener('keydown', function (e) { if (e.key === 'Escape') setMenu(false); });
    $all('nav.main a').forEach(function (a) { a.addEventListener('click', function () { setMenu(false); }); });
  }

  // ---------- platform detection and download links ----------
  var TARGETS = {
    'linux-x64':     { os: 'Linux',   label: 'Linux (x64)',            file: 'jax' },
    'linux-arm64':   { os: 'Linux',   label: 'Linux (arm64)',          file: 'jax' },
    'macos-arm64':   { os: 'macOS',   label: 'macOS (Apple Silicon)',  file: 'jax' },
    'macos-x64':     { os: 'macOS',   label: 'macOS (Intel)',          file: 'jax' },
    'windows-x64':   { os: 'Windows', label: 'Windows (x64)',          file: 'jax.exe' },
    'windows-arm64': { os: 'Windows', label: 'Windows (arm64)',        file: 'jax.exe' }
  };
  var state = { os: null, arch: 'x64', version: '0.1.0' };
  var ALL_DOWNLOADS = SPA ? '#/download/builds' : ROOT + 'index.html#/download/builds';

  function guess() {
    var ua = navigator.userAgent || '';
    var plat = (navigator.userAgentData && navigator.userAgentData.platform) || navigator.platform || '';
    var os = null;
    if (/Android|iPhone|iPad|iPod/i.test(ua)) os = 'mobile';
    else if (/Win/i.test(plat) || /Windows/i.test(ua)) os = 'windows';
    else if (/Mac/i.test(plat) || /Macintosh|Mac OS X/i.test(ua)) os = (navigator.maxTouchPoints > 1 ? 'mobile' : 'macos');
    else if (/Linux|X11/i.test(plat + ' ' + ua)) os = 'linux';
    state.os = os;
    // Macs sold since 2020 are Apple Silicon; browsers other than Chromium cannot tell us more.
    state.arch = os === 'macos' ? 'arm64' : 'x64';
    if (/aarch64|arm64|armv8/i.test(ua + ' ' + plat)) state.arch = 'arm64';
  }

  function targetKey() {
    if (!state.os || state.os === 'mobile') return null;
    var key = state.os + '-' + state.arch;
    return TARGETS[key] ? key : null;
  }

  function fill(key) {
    var t = key ? TARGETS[key] : null;
    var fallbackKey = key || 'linux-x64';
    var f = TARGETS[fallbackKey];
    var base = SITE + '/dist/bin/' + fallbackKey + '/';

    $all('[data-dl]').forEach(function (a) {
      a.setAttribute('href', t ? ROOT + 'dist/bin/' + key + '/' + t.file : ALL_DOWNLOADS);
    });
    $all('[data-dl-label]').forEach(function (el) {
      el.textContent = t ? 'Download for ' + t.os : 'See all downloads';
    });
    $all('[data-fill]').forEach(function (el) {
      var k = el.getAttribute('data-fill');
      var v = '';
      if (k === 'platform') v = t ? t.label : 'Linux, macOS or Windows';
      else if (k === 'target') v = fallbackKey;
      else if (k === 'file') v = f.file;
      else if (k === 'sha') v = f.file + '.sha256';
      else if (k === 'url') v = base + f.file;
      else if (k === 'shaurl') v = base + f.file + '.sha256';
      else if (k === 'version') v = state.version;
      el.textContent = v;
    });
    $all('[data-row]').forEach(function (tr) {
      var yours = !!key && tr.getAttribute('data-row') === key;
      tr.classList.toggle('is-yours', yours);
      var cell = tr.querySelector('td.os');
      var tag = cell && cell.querySelector('.you');
      if (yours && cell && !tag) {
        var s = doc.createElement('span');
        s.className = 'you';
        s.textContent = 'Your computer';
        cell.insertBefore(s, cell.querySelector('small'));
      } else if (!yours && tag) { tag.parentNode.removeChild(tag); }
    });
    $all('.install').forEach(function (box) {
      if (box.getAttribute('data-picked')) return;
      var tabs = $all('[role=tab]', box);
      if (tabs.length) selectTab(box, state.os === 'windows' ? 1 : 0);
    });
  }

  function selectTab(box, i) {
    var tabs = $all('[role=tab]', box), panels = $all('.panel', box);
    tabs.forEach(function (t, j) { t.setAttribute('aria-selected', j === i ? 'true' : 'false'); });
    panels.forEach(function (p, j) { p.classList.toggle('on', j === i); });
  }
  $all('.install').forEach(function (box) {
    $all('[role=tab]', box).forEach(function (t, i) {
      t.addEventListener('click', function () { box.setAttribute('data-picked', '1'); selectTab(box, i); });
    });
  });

  // after a download starts, tell the person what to do next
  $all('.manifest [data-dl]').forEach(function (a) {
    a.addEventListener('click', function () {
      if (targetKey()) a.closest('.manifest').classList.add('sent');
    });
  });

  guess();
  fill(targetKey());

  if (navigator.userAgentData && navigator.userAgentData.getHighEntropyValues && state.os && state.os !== 'mobile') {
    navigator.userAgentData.getHighEntropyValues(['architecture']).then(function (v) {
      var a = String(v.architecture || '').toLowerCase();
      if (a.indexOf('arm') === 0) state.arch = 'arm64';
      else if (a.indexOf('x86') === 0) state.arch = 'x64';
      fill(targetKey());
    }).catch(noop);
  }

  // the newest toolchain version the server offers
  if (window.fetch) {
    fetch(ROOT + 'dist/latest.txt', { cache: 'no-cache' }).then(function (r) {
      return r.ok ? r.text() : Promise.reject();
    }).then(function (txt) {
      var v = String(txt).trim().split(/\s+/)[0];
      if (/^\d+\.\d+\.\d+([-+.][0-9A-Za-z.-]+)?$/.test(v)) { state.version = v; fill(targetKey()); }
    }).catch(noop);
  }

  // ---------- copy buttons ----------
  function copyText(text, btn) {
    var done = function () { var old = btn.textContent; btn.textContent = 'Copied'; setTimeout(function () { btn.textContent = old; }, 1400); };
    if (navigator.clipboard && navigator.clipboard.writeText) { navigator.clipboard.writeText(text).then(done, done); }
    else {
      var ta = doc.createElement('textarea'); ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
      doc.body.appendChild(ta); ta.select();
      try { doc.execCommand('copy'); } catch (e) {}
      doc.body.removeChild(ta); done();
    }
  }
  function commandsFrom(text) {
    text = text.replace(/\s+$/, '');
    if (/^\$ /m.test(text)) {
      return text.split('\n').filter(function (l) { return /^\$ /.test(l); }).map(function (l) { return l.slice(2); }).join('\n');
    }
    return text;
  }

  $all('.cmd .copy').forEach(function (b) {
    b.addEventListener('click', function () {
      // hidden views have no layout, so read textContent rather than innerText
      copyText(b.parentElement.querySelector('code').textContent.trim(), b);
    });
  });

  $all('article pre, .prose pre').forEach(function (pre) {
    var code = pre.querySelector('code');
    if (!code) return;
    var wrap = doc.createElement('div');
    wrap.className = 'codeblock';
    pre.parentNode.insertBefore(wrap, pre);
    wrap.appendChild(pre);
    var b = doc.createElement('button');
    b.type = 'button'; b.className = 'copy'; b.textContent = 'Copy';
    b.setAttribute('aria-label', 'Copy code');
    b.addEventListener('click', function () { copyText(commandsFrom(code.textContent), b); });
    wrap.appendChild(b);
  });

  // ---------- Jax syntax highlighter ----------
  var KW = 'fn|let|var|const|if|else|match|while|loop|for|in|return|break|continue|struct|enum|impl|trait|pub|use|as|true|false|none|self|mut|with|test|async|await|move|some';
  var re = new RegExp('(\\/\\/[^\\n]*)|("(?:[^"\\\\]|\\\\.)*")|\\b(' + KW + ')\\b|\\b([A-Z][A-Za-z0-9_]*)\\b|\\b(\\d[\\d_]*(?:\\.\\d+)?)\\b', 'g');
  $all('pre code.jax').forEach(function (el) {
    var src = el.textContent, out = '', last = 0, m;
    re.lastIndex = 0;
    while ((m = re.exec(src))) {
      out += esc(src.slice(last, m.index));
      var cls = m[1] ? 'c' : m[2] ? 's' : m[3] ? 'k' : m[4] ? 't' : 'n';
      out += '<span class="' + cls + '">' + esc(m[0]) + '</span>';
      last = re.lastIndex;
    }
    el.innerHTML = out + esc(src.slice(last));
  });

  if (!SPA) return;

  // ---------- docs: heading anchors and "on this page" ----------
  views.forEach(function (view) {
    var route = view.getAttribute('data-route');
    var key = keyOf(route);
    var heads = $all('article h2, .prose h2', view);
    if (!heads.length) return;
    var used = {};
    var items = heads.map(function (h) {
      var text = h.textContent.trim();
      var short;
      if (h.id) { short = h.id.indexOf('--') > -1 ? h.id.split('--').slice(1).join('--') : h.id; }
      else {
        short = text.toLowerCase().replace(/&/g, 'and').replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'section';
        while (used[short]) short += '-2';
      }
      used[short] = 1;
      h.id = key + '--' + short;
      var a = doc.createElement('a');
      a.className = 'anchor'; a.href = '#/' + route + '/' + short; a.textContent = '#';
      a.setAttribute('aria-label', 'Link to ' + text);
      h.appendChild(a);
      return { short: short, text: text, el: h };
    });
    var side = view.querySelector('.side');
    if (side && view.querySelector('article') && items.length > 2) {
      var toc = doc.createElement('div');
      toc.className = 'toc';
      toc.innerHTML = '<h4>On this page</h4>' + items.map(function (i) {
        return '<a href="#/' + route + '/' + i.short + '" data-short="' + i.short + '">' + esc(i.text) + '</a>';
      }).join('');
      side.appendChild(toc);
      var links = $all('a', toc);
      if ('IntersectionObserver' in window) {
        var io = new IntersectionObserver(function (entries) {
          entries.forEach(function (e) {
            if (e.isIntersecting) {
              var short = e.target.id.split('--').slice(1).join('--');
              links.forEach(function (l) { l.classList.toggle('on', l.getAttribute('data-short') === short); });
            }
          });
        }, { rootMargin: '-90px 0px -70% 0px' });
        items.forEach(function (i) { io.observe(i.el); });
      }
    }
  });

  // ---------- router: one page, many views ----------
  var current = null;
  function show() {
    var p = window.jaxParse(location.hash);
    if (!p) return; // a same-page target such as #main: leave the view alone
    var changed = p.route !== current;
    current = p.route;
    root.setAttribute('data-route', p.route);
    var view = doc.querySelector('.view[data-route="' + p.route + '"]');
    if (view) doc.title = view.getAttribute('data-title');
    $all('nav.main a').forEach(function (a) {
      var on = a.getAttribute('href') === '#/' + p.route;
      if (on) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
    });
    setMenu(false);
    var target = p.anchor && doc.getElementById(keyOf(p.route) + '--' + p.anchor);
    if (target) target.scrollIntoView({ behavior: 'instant', block: 'start' });
    else if (changed || !p.anchor) window.scrollTo({ top: 0, left: 0, behavior: 'instant' });
  }
  window.addEventListener('hashchange', show);
  show();

  // the skip link jumps to main without changing the view
  var skip = doc.querySelector('.skip');
  if (skip) skip.addEventListener('click', function (e) {
    e.preventDefault();
    var m = doc.getElementById('main');
    if (m) { m.focus({ preventScroll: true }); m.scrollIntoView({ behavior: 'instant' }); }
  });
})();
