// Headless WebSDR (PA3FWM) client: opens the WebSDR page in headless
// Firefox (the system one, driven over WebDriver BiDi, so no Playwright
// browser download is needed; FIREFOX overrides its path), taps the decoded audio through the page's hook_webaud hook,
// and writes it to stdout as s16le mono resampled to 8000 Hz.
// usage: node websdr_bridge.js [URL]   (status lines on stderr)
// The page stops its own audio after the site's idle timeout (2 hours on
// the Utah WebSDR, bandinfo.js idletimeout) while the websocket stays
// open. With STALL_S set (seconds; off by default), the page is loaded
// again when no samples arrive for that long, which defeats that
// timeout: the Northern Utah WebSDR's terms allow it only with the
// operators' explicit permission.
// WEBSDR_NAME (a callsign) is given to the site as the listener's name,
// and once audio flows the page switches to its "blind" view, which
// closes the waterfall stream.
const { _bidiFirefox: firefox } = require('playwright');
const url = process.argv[2] || 'http://websdr1.sdrutah.org:8901/index1a.html?tune=7066usb';
const OUT = 8000;
const STALL_S = Number(process.env.STALL_S || 0);
const NAME = (process.env.WEBSDR_NAME || '').replace(/[^ -~]/g, '').slice(0, 16);
(async () => {
  const browser = await firefox.launch({
    executablePath: process.env.FIREFOX || '/usr/bin/firefox',
    // Autoplay without a gesture; output muted (the hook taps the audio).
    // Background services (telemetry, updates, connectivity checks) off.
    firefoxUserPrefs: {
      'media.autoplay.default': 0, 'media.autoplay.blocking_policy': 0, 'media.volume_scale': '0.0',
      'datareporting.policy.dataSubmissionEnabled': false, 'datareporting.healthreport.uploadEnabled': false,
      'toolkit.telemetry.enabled': false, 'app.update.enabled': false, 'app.update.auto': false,
      'network.connectivity-service.enabled': false, 'network.captive-portal-service.enabled': false,
      'browser.safebrowsing.malware.enabled': false, 'browser.safebrowsing.phishing.enabled': false,
    },
  });
  const page = await browser.newPage();
  let rate = 0, n = 0, pos = 0, last = 0, lastAudio = 0; const t0 = Date.now();
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
    lastAudio = Date.now();
    process.stdout.write(Buffer.from(new Int16Array(out).buffer));
  });
  // The page sends its username cookie with the tuning to the server.
  await page.addInitScript(name => {
    if (name && /^https?:/.test(location.protocol)) document.cookie = `username=${name}; path=/; SameSite=Lax`;
  }, NAME);
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
  // Load the page and press Firefox's audio start button until audio
  // arrives (Firefox gets none until it is pressed). The init script and
  // pushAudio carry over to every load.
  const connect = async () => {
    const before = n;
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });
    const start = page.locator('input[onclick^="mozilla_audio_start"]');
    for (let i = 0; i < 10 && n === before; i++) {
      await start.click({ timeout: 10000 }).catch(e => process.stderr.write(`audio start: ${e.message.split('\n')[0]}\n`));
      await new Promise(r => setTimeout(r, 3000));
    }
    if (n === before) process.stderr.write('no audio after pressing audio start\n');
    // Switch to the "blind" view, which closes the waterfall stream. Not
    // via the view cookie: on the Utah page a view=3 cookie makes
    // bodyonload throw before the audio client starts. setview() saves
    // view=3 itself, so put the cookie back for the next load.
    else await page.evaluate(() => { setview(3); document.cookie = 'view=2; path=/; SameSite=Lax'; })
      .catch(e => process.stderr.write(`blind view: ${e.message.split('\n')[0]}\n`));
    lastAudio = Date.now();
  };
  await connect();
  // Watchdog: reload when the audio has stalled; back off while reloads
  // keep failing (server down, network gone).
  let busy = false, fails = 0, next = 0;
  if (STALL_S > 0) setInterval(async () => {
    const now = Date.now();
    if (busy || now < next || now - lastAudio < STALL_S * 1000) return;
    busy = true;
    const iso = new Date(now).toISOString();
    process.stderr.write(`${iso} no audio for ${((now - lastAudio) / 1000).toFixed(0)} s at ${n} samples; reconnecting\n`);
    const before = n;
    try { await connect(); } catch (e) { process.stderr.write(`reconnect: ${e.message.split('\n')[0]}\n`); }
    if (n > before) { fails = 0; process.stderr.write(`${new Date().toISOString()} reconnected\n`); }
    else { fails++; next = Date.now() + Math.min(300, 15 * 2 ** fails) * 1000; }
    busy = false;
  }, 5000);
  setInterval(() => {
    const secs = (Date.now() - t0) / 1000;
    process.stderr.write(`out ${n} samples in ${secs.toFixed(0)} s (${(n / secs).toFixed(0)}/s)\n`);
  }, 60000);
})().catch(e => { console.error(e); process.exit(1); });
