// Headless WebSDR (PA3FWM) client: opens the WebSDR page in headless
// Firefox (the system one, driven over WebDriver BiDi, so no Playwright
// browser download is needed; FIREFOX overrides its path), taps the decoded audio through the page's hook_webaud hook,
// and writes it to stdout as s16le mono resampled to 8000 Hz.
// usage: node websdr_bridge.js [URL]   (status lines on stderr)
const { _bidiFirefox: firefox } = require('playwright');
const url = process.argv[2] || 'http://websdr1.sdrutah.org:8901/index1a.html?tune=7066usb';
const OUT = 8000;
(async () => {
  const browser = await firefox.launch({
    executablePath: process.env.FIREFOX || '/usr/bin/firefox',
    // Autoplay without a gesture; output muted (the hook taps the audio).
    firefoxUserPrefs: { 'media.autoplay.default': 0, 'media.autoplay.blocking_policy': 0, 'media.volume_scale': '0.0' },
  });
  const page = await browser.newPage();
  let rate = 0, n = 0, pos = 0, last = 0; const t0 = Date.now();
  await page.exposeFunction('pushAudio', (b64, D) => {
    const b = Buffer.from(b64, 'base64');
    const x = new Int16Array(b.buffer, b.byteOffset, b.length / 2);
    if (D !== rate) { rate = D; process.stderr.write(`websdr rate ${D} Hz\n`); }
    const step = D / OUT, out = [];
    // linear interpolation across chunk boundaries (pos is in input samples)
    while (pos < x.length) {
      const i = Math.floor(pos), f = pos - i;
      const a = i < 0 ? last : x[i], c = i + 1 < x.length ? x[i + 1] : x[i];
      out.push(Math.round(a + (c - a) * f)); pos += step;
    }
    pos -= x.length; last = x[x.length - 1];
    n += out.length;
    process.stdout.write(Buffer.from(new Int16Array(out).buffer));
  });
  await page.addInitScript(() => {
    window.webaud_hookdef = 1;
    const q = []; let D = 0;
    const hook = (m, d) => { q.push(m); D = d; return m; };
    // The page's websdr-soundhooks.js assigns its own hook_webaud (and
    // webaud_hookdef=0, which still counts as defined); keep ours.
    Object.defineProperty(window, 'hook_webaud', { get: () => hook, set: () => {}, configurable: false });
    setInterval(() => {
      if (!q.length) return;
      const u = new Uint8Array(new Int16Array(q.splice(0)).buffer);
      let s = '';
      for (let i = 0; i < u.length; i += 8192) s += String.fromCharCode.apply(null, u.subarray(i, i + 8192));
      window.pushAudio(btoa(s), D);
    }, 200);
  });
  page.on('close', () => { process.stderr.write('page closed\n'); process.exit(2); });
  page.on('crash', () => process.stderr.write('page crashed\n'));
  browser.on('disconnected', () => process.stderr.write('browser disconnected\n'));
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });
  // Firefox gets no audio until the page's "Firefox/Mozilla audio start"
  // button is pressed; press it until audio arrives.
  const start = page.locator('input[onclick^="mozilla_audio_start"]');
  for (let i = 0; i < 10 && !rate; i++) {
    await start.click({ timeout: 10000 }).catch(e => process.stderr.write(`audio start: ${e.message.split('\n')[0]}\n`));
    await new Promise(r => setTimeout(r, 3000));
  }
  if (!rate) process.stderr.write('no audio after pressing audio start\n');
  setInterval(() => {
    const secs = (Date.now() - t0) / 1000;
    process.stderr.write(`out ${n} samples in ${secs.toFixed(0)} s (${(n / secs).toFixed(0)}/s)\n`);
  }, 60000);
})().catch(e => { console.error(e); process.exit(1); });
