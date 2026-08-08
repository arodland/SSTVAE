package org.cleverdomain.sstvae;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.FileOutputStream;
import java.net.HttpURLConnection;
import java.net.URL;

/**
 * HTTPS transport for the model download, and nothing else.
 *
 * <p><b>Why not QtNetwork.</b> Qt for Android ships no TLS backend —
 * {@code qt.tlsbackend.ossl: Failed to load libssl/libcrypto} — so
 * using it means bundling OpenSSL per ABI and owning its patch cadence.
 * {@code HttpsURLConnection} uses the platform stack and the system
 * trust store instead, which is the same reasoning that chose Java
 * {@code AudioRecord} over QtMultimedia: on the two occasions this port
 * has met an Android platform service, going through the platform has
 * been both smaller and more correct.
 *
 * <p><b>This class deliberately does not verify anything.</b> It
 * streams bytes to a {@code .part} file and hands back the sha256 the
 * Hub declared; the comparison, and the rename that makes a file
 * visible to {@code find_cached}, both stay in C++. That is the whole
 * point of the split — the security-relevant step keeps one
 * implementation, shared with the desktop's fetcher, and this side
 * cannot get it wrong because it does not do it.
 *
 * <p>Redirects are followed <b>by hand</b>, for the same reason
 * {@code qt_fetcher.cpp} does: the Hub's 302 carries
 * {@code x-linked-etag}, the LFS object's sha256, and automatic
 * following would consume the response that states it. The checksum has
 * to come from the server <i>before</i> the bytes it describes.
 */
public final class ModelFetcher {
    private static final int MAX_REDIRECTS = 5;
    private static final int TIMEOUT_MS = 30000;

    private ModelFetcher() {}

    private static native void nativeProgress(long received, long total);

    /**
     * Download {@code url} into {@code partPath}, following redirects.
     *
     * @return the sha256 hex the Hub declared for the object, or an
     *     empty string if it declared none. Never null.
     * @throws IOException on any transport failure, with a message
     *     meant for an operator rather than a log.
     */
    public static String download(String url, String partPath) throws IOException {
        String linkedSha = "";
        String current = url;

        for (int hop = 0; ; hop++) {
            if (hop > MAX_REDIRECTS) {
                throw new IOException("too many redirects fetching " + url);
            }
            HttpURLConnection conn = (HttpURLConnection) new URL(current).openConnection();
            conn.setInstanceFollowRedirects(false);
            conn.setConnectTimeout(TIMEOUT_MS);
            conn.setReadTimeout(TIMEOUT_MS);
            try {
                final int status = conn.getResponseCode();

                // Read it on every hop, not just the first: it is the
                // redirect that carries it, and which hop that is
                // depends on the Hub's CDN.
                String etag = conn.getHeaderField("x-linked-etag");
                if (etag != null && !etag.isEmpty()) {
                    linkedSha = etag.replace("\"", "");
                }

                if (status == HttpURLConnection.HTTP_MOVED_PERM
                        || status == HttpURLConnection.HTTP_MOVED_TEMP
                        || status == HttpURLConnection.HTTP_SEE_OTHER
                        || status == 307
                        || status == 308) {
                    final String next = conn.getHeaderField("Location");
                    if (next == null || next.isEmpty()) {
                        throw new IOException("redirect with no Location fetching " + url);
                    }
                    current = new URL(new URL(current), next).toString();
                    continue;
                }
                if (status != HttpURLConnection.HTTP_OK) {
                    throw new IOException("HTTP " + status + " fetching " + url);
                }

                final long total = conn.getContentLengthLong();
                long received = 0;
                byte[] buf = new byte[64 * 1024];
                try (InputStream in = conn.getInputStream();
                     OutputStream out = new FileOutputStream(partPath)) {
                    int n;
                    while ((n = in.read(buf)) > 0) {
                        out.write(buf, 0, n);
                        received += n;
                        nativeProgress(received, total);
                    }
                }
                if (received == 0) {
                    throw new IOException("empty response fetching " + url);
                }
                return linkedSha;
            } finally {
                conn.disconnect();
            }
        }
    }
}
