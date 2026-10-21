(function () {
  // Install tabs
  document.querySelectorAll('.install').forEach(function (box) {
    var tabs = box.querySelectorAll('[role=tab]');
    var panels = box.querySelectorAll('.panel');
    function select(i) {
      tabs.forEach(function (t, j) { t.setAttribute('aria-selected', j === i ? 'true' : 'false'); });
      panels.forEach(function (p, j) { p.classList.toggle('on', j === i); });
    }
    tabs.forEach(function (t, i) { t.addEventListener('click', function () { select(i); }); });
    var ua = navigator.userAgent || '';
    select(/Windows/i.test(ua) ? 1 : 0);
  });

  // Copy buttons
  document.querySelectorAll('.copy').forEach(function (b) {
    b.addEventListener('click', function () {
      var text = b.parentElement.querySelector('code').innerText.trim();
      var done = function () { var old = b.textContent; b.textContent = 'Copied'; setTimeout(function () { b.textContent = old; }, 1400); };
      if (navigator.clipboard) { navigator.clipboard.writeText(text).then(done, done); } else { done(); }
    });
  });

  // Tiny Jax syntax highlighter
  var KW = 'fn|let|var|const|if|else|match|while|loop|for|in|return|break|continue|struct|enum|impl|trait|pub|use|as|true|false|none|self|mut|with|test|async|await|move|some';
  var re = new RegExp('(\\/\\/[^\\n]*)|("(?:[^"\\\\]|\\\\.)*")|\\b(' + KW + ')\\b|\\b([A-Z][A-Za-z0-9_]*)\\b|\\b(\\d[\\d_]*(?:\\.\\d+)?)\\b', 'g');
  function esc(s) { return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
  document.querySelectorAll('pre code.jax').forEach(function (el) {
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
})();
